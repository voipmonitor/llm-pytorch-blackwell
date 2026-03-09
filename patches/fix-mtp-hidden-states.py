"""Fix MTP hidden_states regression from vLLM PR #34552.

PR #34552 simplified the MTP method check from an explicit list of model types
to `self.method == "mtp"`, inadvertently activating a previously dead code path
that feeds cached target hidden states instead of model output to subsequent
MTP layers. This causes ~30% acceptance rate regression for chained-MTP models
like Qwen3.5.

Upstream fix: https://github.com/vllm-project/vllm/pull/36531
Remove this patch once merged.
"""
import vllm.v1.spec_decode.eagle

f = vllm.v1.spec_decode.eagle.__file__
c = open(f).read()

old = (
    '        if self.method == "mtp":\n'
    '            hidden_states = self.hidden_states[token_indices_to_sample]\n'
    '        else:\n'
    '            hidden_states = hidden_states[token_indices_to_sample]'
)

new = (
    '        # Fix: propagate model output for chained MTP (Qwen3.5).\n'
    '        # See PR #34552 regression, fix: PR #36531\n'
    '        hidden_states = hidden_states[token_indices_to_sample]'
)

assert old in c, 'Pattern not found — maybe already fixed upstream. Remove this patch.'
c = c.replace(old, new)
open(f, 'w').write(c)
print('OK: MTP hidden_states fix applied to', f)
