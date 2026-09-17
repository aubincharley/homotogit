# Verbatim author code used only for verification

Byte copies (MIT licence, licence files included) of the files read for the port. Never imported by
training code; the tests import `cbs_5f62e7d/utils.py` and `sdpoint_0013c5d/models/resnet.py` to
check kernels, forwards and the two SDPoint selection defects.

| file | repository @ commit | sha256 |
|---|---|---|
| cbs_5f62e7d/utils.py | pairlab/CBS @ 5f62e7da5b290e8f62f408c6f89146f3f361cc2e | 0a16ec1412940b01833e9e8ec9fa657db6b8444019525fe1fe3d276e02405684 |
| cbs_5f62e7d/resnet.py | same | 768472846c14318fdb8eacc0cd2fb48b11c253f49c71b2b88c5b410b70d90ad2 |
| cbs_5f62e7d/solver_cbs.py | same | 1005545f460ddd8777f61f14a2309efc31afa597575fb8ec85e995ee05231635 |
| cbs_5f62e7d/arguments.py | same | 36e8e60eb1904534bd3c617f7d882dac0e9788bb42d483a368f7a10ab93e2041 |
| sdpoint_0013c5d/models/resnet.py | xternalz/SDPoint @ 0013c5dafe80780ea749198ebc42824c3ed41e6c | 147a42828b2094c05000d5f32fc348d1bc9c88a45929b10c0fb7eed0ec60c2c5 |
| sdpoint_0013c5d/models/preresnet.py | same | 7dfc2b54f70beb9018b867001d56f81ad42bb344f77992338b0567056479c718 |
| sdpoint_0013c5d/main.py | same | bb9472a5113d862037beb5985963120ca910f10127194833c9ab2431c65e2b68 |

Also read but not copied: CBS `solver_base.py` (19452578…), `data.py` (605999eb…), `main.py`.
