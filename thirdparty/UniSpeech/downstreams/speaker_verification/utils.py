import os
import torch
import fairseq
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from fairseq import tasks
from torch import Tensor
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from speaker_verification.WavLM import WavLM, WavLMConfig
from typing import Callable, List, Tuple


SAMPLE_RATE = 16000
TOLERABLE_SEQLEN_DIFF = 5


class Hook:
    def __init__(self, module_path, transform, unique_identifier=None):
        self.module_path = module_path
        self.transform = transform
        self.unique_identifier = unique_identifier or module_path
        self.handler = None

        assert isinstance(self.module_path, str)
        assert callable(self.transform)
        assert isinstance(self.unique_identifier, str)


class initHook(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        for hook in instance.hooks:
            if hook.handler is None:
                instance._register_hook_handler(hook)
        return instance


class UpstreamBase(nn.Module, metaclass=initHook):
    def __init__(
        self,
        hooks: List[Tuple] = None,
        hook_postprocess: Callable[
            [List[Tuple[str, Tensor]]], List[Tuple[str, Tensor]]
        ] = None,
        **kwargs,
    ):
        """
        Args:
            hooks: each Tuple is an argument list for the Hook initializer
        """
        super().__init__()
        self.hooks: List[Hook] = [Hook(*hook) for hook in hooks] if hooks else []
        self.hook_postprocess = hook_postprocess
        self._hook_hiddens: List[Tuple(str, Tensor)] = []

    def remove_all_hooks(self):
        for hook in self.hooks:
            hook.handler.remove()
        self.hooks.clear()

    def remove_hook(self, unique_identifier: str):
        updated_hooks = []
        for hook in self.hooks:
            if hook.unique_identifier == unique_identifier:
                hook.handler.remove()
            else:
                updated_hooks.append(hook)
        self.hooks = updated_hooks

    def add_hook(self, *args, **kwargs):
        hook = Hook(*args, **kwargs)
        self._register_hook_handler(hook)
        self.hooks.append(hook)

    def _register_hook_handler(self, hook: Hook):
        module = eval(hook.module_path)
        if not isinstance(module, nn.Module):

            return

        if callable(hook.handler):

            hook.handler.remove()

        def generate_hook_handler(hiddens: List, hook: Hook):
            def hook_handler(self, input, output):
                hiddens.append((hook.unique_identifier, hook.transform(input, output)))

            return hook_handler

        hook.handler = module.register_forward_hook(
            generate_hook_handler(self._hook_hiddens, hook)
        )

    def __call__(self, wavs: List[Tensor], *args, **kwargs):
        self._hook_hiddens.clear()

        result = super().__call__(wavs, *args, **kwargs) or {}
        assert isinstance(result, dict)

        if len(self._hook_hiddens) > 0:
            if (
                result.get("_hidden_states_info") is not None
                or result.get("hidden_states") is not None
                or result.get("last_hidden_state") is not None
            ):

                raise ValueError

            hook_hiddens = self._hook_hiddens.copy()
            self._hook_hiddens.clear()

            if callable(self.hook_postprocess):
                hook_hiddens = self.hook_postprocess(hook_hiddens)

            result["_hidden_states_info"], result["hidden_states"] = zip(*hook_hiddens)
            result["last_hidden_state"] = result["hidden_states"][-1]

            for layer_id, hidden_state in enumerate(result["hidden_states"]):
                result[f"hidden_state_{layer_id}"] = hidden_state

        return result


def load_model(filepath):
    state = torch.load(filepath, map_location=lambda storage, loc: storage)
    # state = load_checkpoint_to_cpu(filepath)
    state["cfg"] = OmegaConf.create(state["cfg"])

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state keys = {state.keys()}"
            )

    task = tasks.setup_task(cfg.task)
    if "task_state" in state:
        task.load_state_dict(state["task_state"])

    model = task.build_model(cfg.model)

    return model, cfg, task


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."
        ckpt = os.path.join(ckpt, r"WavLM-Large.pt")
        checkpoint = torch.load(ckpt)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(self.cfg)

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

    def forward(self, wavs):
        if self.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )
        return {
            "default": features,
        }
