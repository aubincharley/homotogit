# Checkpoint retrieval manifest

Large checkpoints are not in the handoff. New runs: Kaggle kernel output `<user>/ovn-<tag>-<account>` under `ovn/cells/<run>/checkpoints/epoch_{030,060,120,160}.pt`, mirrored locally under `studies/overnight_long_aug_bn/raw/<tag>/<account>/ovn/cells/<run>/` (git-ignored). Historical 30-epoch endpoints: dataset `maxmonstre/overnight-inputs`, `old/<run>/epoch_030.pt`. Verify with `sha256`; the digests below are those recorded when each checkpoint was evaluated.

| run | kernel output | epoch-160 checkpoint sha256 |
|---|---|---|
| adamw_long_aug_160__cbs_budget_matched__seed2 | `maxmonstre/ovn-t2-maxmonstre` | `df4a5d346d946266b076e2160c33531e3cb80fb165c4d327e7a98bcb6889bc04` |
| adamw_long_aug_160__resolution_max_b1__seed1 | `maxmonstre/ovn-t2-maxmonstre` | `27226579a036867b509d8ce319139d303c373ec78a32c3ac31bfb3d3678bf9b1` |
| adamw_long_aug_160__resolution_max_b1_gaussian_conv__seed1 | `maxmonstre/ovn-t2-maxmonstre` | `92a169e29e33b06fae441eec03b7021034394eeaf6ad880f84eec1a07af14b14` |
| adamw_long_noaug_160__gaussian_postrelu__seed1 | `maxmonstre/ovn-t2-maxmonstre` | `26ff99eca2324d121a021fb6fe82bb1a9ed7f6d4b4f09bf26aa3cd805cfa3a5a` |
| adamw_long_noaug_160__gaussian_postrelu__seed2 | `maxmonstre/ovn-t2-maxmonstre` | `62613592bb5555f0335ec67feecdcfcf9c7184c6fb46c6f974f7a50384c6efb2` |
| adamw_long_noaug_160__sdpoint__seed0 | `maxmonstre/ovn-t2-maxmonstre` | `f2b11d4b9051e1a96f27987762c27964a1e7b05dab1698ed10d3699c4b7dc4cc` |
| adamw_long_noaug_160__sdpoint__seed2 | `maxmonstre/ovn-t2-maxmonstre` | `1db78922838cf3e50a5b29fb1cece7efa6b2d702b7a0a563df4975d0c1985204` |
| sgd_standard_aug_160__cbs_budget_matched__seed2 | `maxmonstre/ovn-t2-maxmonstre` | `f2aad06a960669d937442ecf810ae92089e9e1974fb304e4f753b5dd7ee728fc` |
| sgd_standard_aug_160__sdpoint__seed2 | `maxmonstre/ovn-t2-maxmonstre` | `beb06fd82d432187fdc5373e45770270c40ef20d1450db3a279610a9858591c4` |

| historical run | location | epoch-30 checkpoint sha256 |
|---|---|---|
| adamw__cbs_budget_matched__seed0 | `maxmonstre/overnight-inputs: old/adamw__cbs_budget_matched__seed0/epoch_030.pt` | `7bf02b8d7396ad35d982e44c956d43559e926f2a5252f1aa5071a0c82453316e` |
| adamw__cbs_budget_matched__seed1 | `maxmonstre/overnight-inputs: old/adamw__cbs_budget_matched__seed1/epoch_030.pt` | `57cc7c24e1ddd0bca397f98e0cc3b5b2ea8c81937ff3c79aa092fb9a3a261632` |
| adamw__cbs_budget_matched__seed2 | `maxmonstre/overnight-inputs: old/adamw__cbs_budget_matched__seed2/epoch_030.pt` | `eecc837d4913e60bd6070e7cf707c69700606feef79c0de42bbb7f634051db62` |
| adamw__cbs_published_schedule__seed0 | `maxmonstre/overnight-inputs: old/adamw__cbs_published_schedule__seed0/epoch_030.pt` | `acf17826041e5caffb495081bd0535b0ec0fb0226dfc32db25c5569fe003436a` |
| adamw__cbs_published_schedule__seed1 | `maxmonstre/overnight-inputs: old/adamw__cbs_published_schedule__seed1/epoch_030.pt` | `f076148c41843f0211ce742aac20b8012301f2d227e08a9625683d259140c9ba` |
| adamw__cbs_published_schedule__seed2 | `maxmonstre/overnight-inputs: old/adamw__cbs_published_schedule__seed2/epoch_030.pt` | `c4eb32ba86b39e4f86139f6ffb6f17c63b24ced1f8f554a65b14c282afff0bd9` |
| adamw__gaussian_postrelu__seed0 | `maxmonstre/overnight-inputs: old/adamw__gaussian_postrelu__seed0/epoch_030.pt` | `9eb7b901911e4a45434deb26c34106bc4d4ba428efa2e5938ed2e8ff82b3c5fb` |
| adamw__gaussian_postrelu__seed1 | `maxmonstre/overnight-inputs: old/adamw__gaussian_postrelu__seed1/epoch_030.pt` | `d3b77ef641b98e16f2417b0e3d1c76f24a2498858ece7b20e43ec2cb9f3e6f19` |
| adamw__gaussian_postrelu__seed2 | `maxmonstre/overnight-inputs: old/adamw__gaussian_postrelu__seed2/epoch_030.pt` | `7acd59744471bd6a104b1fe5d45141e3948cc87b2159c2cf4de1a0b71a752e80` |
| adamw__plain__seed0 | `maxmonstre/overnight-inputs: old/adamw__plain__seed0/epoch_030.pt` | `8683907ab48938baf614455e94943023b0eed668bcb325fee19ed621b9c9d609` |
| adamw__plain__seed1 | `maxmonstre/overnight-inputs: old/adamw__plain__seed1/epoch_030.pt` | `336abca0c7737a6303f51ee0421671d89a1cc0ab182a2dd2ec4c31d0e261318b` |
| adamw__plain__seed2 | `maxmonstre/overnight-inputs: old/adamw__plain__seed2/epoch_030.pt` | `58b5aeb928ee8595676aa870a507e0c01c2513ae74739553387564573cf4b8c3` |
| adamw__resolution_max_b1__seed0 | `maxmonstre/overnight-inputs: old/adamw__resolution_max_b1__seed0/epoch_030.pt` | `38e9230714c5f5b7c434956ad7c8f55ee15c9ddccef3a3774318ddae995cb489` |
| adamw__resolution_max_b1__seed1 | `maxmonstre/overnight-inputs: old/adamw__resolution_max_b1__seed1/epoch_030.pt` | `92ddc1593a400d8ba4dad97c38a636964d7d241b7ce010c6323c450724f7b89e` |
| adamw__resolution_max_b1__seed2 | `maxmonstre/overnight-inputs: old/adamw__resolution_max_b1__seed2/epoch_030.pt` | `f39ed5e9bff9fa0643ab73f33355a081a75d7e10e3bf20694b5b8e6203776a78` |
| adamw__resolution_max_b1_gaussian_conv__seed0 | `maxmonstre/overnight-inputs: old/adamw__resolution_max_b1_gaussian_conv__seed0/epoch_030.pt` | `15a26ada3d32ed2cf46fec56eace11b78fed2bf45a0e7550f7d97ecd4a72d5da` |
| adamw__resolution_max_b1_gaussian_conv__seed1 | `maxmonstre/overnight-inputs: old/adamw__resolution_max_b1_gaussian_conv__seed1/epoch_030.pt` | `268f4ce537637397631d4bf7cd9dcd1343029c51a95d9b4b6922a748ad357a9d` |
| adamw__resolution_max_b1_gaussian_conv__seed2 | `maxmonstre/overnight-inputs: old/adamw__resolution_max_b1_gaussian_conv__seed2/epoch_030.pt` | `c0a09cc738f00443f9cc3635c47a827fcc1fca5046753babe094c4fa111606f6` |
| adamw__sdpoint__seed0 | `maxmonstre/overnight-inputs: old/adamw__sdpoint__seed0/epoch_030.pt` | `dcc60257a43b2abe7597cfe35b872f6250db8c66bd0c225215000e553c309049` |
| adamw__sdpoint__seed1 | `maxmonstre/overnight-inputs: old/adamw__sdpoint__seed1/epoch_030.pt` | `d9c5635925a1f5913c31099065375104549f3e764a22f20a25d53410d884d75e` |
| adamw__sdpoint__seed2 | `maxmonstre/overnight-inputs: old/adamw__sdpoint__seed2/epoch_030.pt` | `f97876761c9660beade1bebcad3d3371f4a6da475266adb1b9ee8531dcbb92ea` |
| sgd__cbs_budget_matched__seed0 | `maxmonstre/overnight-inputs: old/sgd__cbs_budget_matched__seed0/epoch_030.pt` | `5037a73b8decfd78f9f8e5f9f28a141548df0e782d1931cda627d2ba187b7301` |
| sgd__cbs_budget_matched__seed1 | `maxmonstre/overnight-inputs: old/sgd__cbs_budget_matched__seed1/epoch_030.pt` | `fa6c0e5d616f3ed8723990d3ba8fc5bb1493104a5bff6bdfa871f4ccd796c01c` |
| sgd__cbs_budget_matched__seed2 | `maxmonstre/overnight-inputs: old/sgd__cbs_budget_matched__seed2/epoch_030.pt` | `ee31f4bc912eb5a529a7a88a16ac69e6d662861d43a69449664531de3942fb19` |
| sgd__cbs_published_schedule__seed0 | `maxmonstre/overnight-inputs: old/sgd__cbs_published_schedule__seed0/epoch_030.pt` | `256079483c6ed35b4222d205a32d14e30234f07ee4894b36d7e5aece10130cc8` |
| sgd__cbs_published_schedule__seed1 | `maxmonstre/overnight-inputs: old/sgd__cbs_published_schedule__seed1/epoch_030.pt` | `94c538bff1338e388a0d8513c725ea8129cfa38a0ec5124806228787e67ebc24` |
| sgd__cbs_published_schedule__seed2 | `maxmonstre/overnight-inputs: old/sgd__cbs_published_schedule__seed2/epoch_030.pt` | `4292d32ae02eb1261c2af7837eb99ceba70ad0e7bdc0477b6f8b6ad22779fce7` |
| sgd__gaussian_postrelu__seed0 | `maxmonstre/overnight-inputs: old/sgd__gaussian_postrelu__seed0/epoch_030.pt` | `ba6a68e37fedfd3ae6e8c60cbbeb0f8525e8866dac9fec3ae8f53d8d519926ea` |
| sgd__gaussian_postrelu__seed1 | `maxmonstre/overnight-inputs: old/sgd__gaussian_postrelu__seed1/epoch_030.pt` | `bbd8e7a5cd3423aec5dff9f15d92708212b64cd4d1a6c0e23b9bec336f2f11d3` |
| sgd__gaussian_postrelu__seed2 | `maxmonstre/overnight-inputs: old/sgd__gaussian_postrelu__seed2/epoch_030.pt` | `3f840c037b78714d81803bc90780c47b3c0ac01fd1a41fa60ada72700aa193b2` |
| sgd__plain__seed0 | `maxmonstre/overnight-inputs: old/sgd__plain__seed0/epoch_030.pt` | `912695496845934775ee1283212a857ceecc718f49b7044a4fdffa3d44d3b324` |
| sgd__plain__seed1 | `maxmonstre/overnight-inputs: old/sgd__plain__seed1/epoch_030.pt` | `9e558ec9f5174e4f3c47df78ca5df5cbe271d4ae95a6aef07a7c3c533b225983` |
| sgd__plain__seed2 | `maxmonstre/overnight-inputs: old/sgd__plain__seed2/epoch_030.pt` | `05c92855a89d14504c941532bc8390c07e6f9d9c700a88e1fd2693ea06693e71` |
| sgd__resolution_max_b1__seed0 | `maxmonstre/overnight-inputs: old/sgd__resolution_max_b1__seed0/epoch_030.pt` | `687d40baae504b1e840be8be92f71dc22b240d8215f28be311693a48873c0bb1` |
| sgd__resolution_max_b1__seed1 | `maxmonstre/overnight-inputs: old/sgd__resolution_max_b1__seed1/epoch_030.pt` | `f49d0ed8412c84bfea0f9655c55e8e3588cda1aae61acb3a8139af3a58ef7443` |
| sgd__resolution_max_b1__seed2 | `maxmonstre/overnight-inputs: old/sgd__resolution_max_b1__seed2/epoch_030.pt` | `6aab6b89559c44074b0b7c6523ccf3854c1a208db35dd03afc6a41302f76db79` |
| sgd__resolution_max_b1_gaussian_conv__seed0 | `maxmonstre/overnight-inputs: old/sgd__resolution_max_b1_gaussian_conv__seed0/epoch_030.pt` | `23208fd45a23aaac69c4d1436d28f740e424a4dccef20df9b302277cbac66151` |
| sgd__resolution_max_b1_gaussian_conv__seed1 | `maxmonstre/overnight-inputs: old/sgd__resolution_max_b1_gaussian_conv__seed1/epoch_030.pt` | `2f8f25cd1950b49c8d7af040978f5cc3c4f3e7f323f545675b5b5e6830e818e6` |
| sgd__resolution_max_b1_gaussian_conv__seed2 | `maxmonstre/overnight-inputs: old/sgd__resolution_max_b1_gaussian_conv__seed2/epoch_030.pt` | `e011741c20d4afeb24572dfcfa34290559932c3ed70d1c8cbfbf8b802e5943fb` |
| sgd__sdpoint__seed0 | `maxmonstre/overnight-inputs: old/sgd__sdpoint__seed0/epoch_030.pt` | `919e8b1d07ec84a4cb51a7f537f9179da713527ea57cdc5be2d57a379a6250a9` |
| sgd__sdpoint__seed1 | `maxmonstre/overnight-inputs: old/sgd__sdpoint__seed1/epoch_030.pt` | `adcd4c8305a2b954a07e9d5f15fa8d5e7b68bf14db5121068628c5d177ef7eb8` |
| sgd__sdpoint__seed2 | `maxmonstre/overnight-inputs: old/sgd__sdpoint__seed2/epoch_030.pt` | `215285755298c1e9ebd42797f86aa9076d44d5d28c62c4fa4298d5a8806dca0a` |
