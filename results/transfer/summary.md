| dataset | protocol | corrector | n | params | test MAE causal | test MAE bidir | gap % | draft MAE |
|---|---|---|---|---|---|---|---|---|
| Beijing PM2.5 (log1p(ug/m3)) | draft_bidir | mamba | 3 | 5370 | 0.5339 ± 0.0194 | 0.4854 ± 0.0018 | +10.0 | 0.6270 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_bidir | rpa | 3 | 4362 | 0.6001 ± 0.0118 | 0.4992 ± 0.0075 | +20.3 | 0.6162 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_bidir | transformer | 3 | 4636 | 0.6276 ± 0.0342 | 0.4943 ± 0.0029 | +27.0 | 0.6316 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_causal | mamba | 3 | 5370 | 0.4804 ± 0.0073 | 0.5090 ± 0.0145 | -5.6 | 0.6127 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_causal | rpa | 3 | 4362 | 0.4886 ± 0.0095 | 0.5149 ± 0.0079 | -5.1 | 0.6304 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_causal | transformer | 3 | 4636 | 0.4871 ± 0.0073 | 0.5810 ± 0.0219 | -16.0 | 0.6391 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_dual | mamba | 3 | 5370 | 0.4864 ± 0.0060 | 0.4884 ± 0.0037 | -0.4 | 0.6094 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_dual | rpa | 3 | 4362 | 0.4943 ± 0.0085 | 0.5029 ± 0.0053 | -1.7 | 0.6151 |
| Beijing PM2.5 (log1p(ug/m3)) | draft_dual | transformer | 3 | 4636 | 0.4916 ± 0.0041 | 0.4968 ± 0.0032 | -1.1 | 0.6107 |
| Beijing PM2.5 (log1p(ug/m3)) | gtmix | mamba | 3 | 5370 | 0.5895 ± 0.0191 | 0.5063 ± 0.0164 | +16.5 | 0.6118 |
| Beijing PM2.5 (log1p(ug/m3)) | gtmix | rpa | 3 | 4362 | 0.5902 ± 0.0263 | 0.5019 ± 0.0045 | +17.6 | 0.6118 |
| Beijing PM2.5 (log1p(ug/m3)) | gtmix | transformer | 3 | 4636 | 0.7320 ± 0.0513 | 0.5820 ± 0.0920 | +27.7 | 0.6642 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_bidir | mamba | 3 | 4538 | 0.2297 ± 0.0054 | 0.2243 ± 0.0021 | +2.4 | 0.2355 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_bidir | rpa | 3 | 3530 | 0.2571 ± 0.0107 | 0.2325 ± 0.0040 | +10.5 | 0.2321 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_bidir | transformer | 3 | 3804 | 0.2463 ± 0.0151 | 0.2204 ± 0.0019 | +11.7 | 0.2279 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_causal | mamba | 3 | 4538 | 0.2269 ± 0.0006 | 0.2245 ± 0.0019 | +1.0 | 0.2341 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_causal | rpa | 3 | 3530 | 0.2289 ± 0.0011 | 0.2257 ± 0.0032 | +1.4 | 0.2285 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_causal | transformer | 3 | 3804 | 0.2268 ± 0.0054 | 0.2295 ± 0.0008 | -1.2 | 0.2321 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_dual | mamba | 3 | 4538 | 0.2308 ± 0.0076 | 0.2302 ± 0.0091 | +0.3 | 0.2295 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_dual | rpa | 3 | 3530 | 0.2292 ± 0.0026 | 0.2277 ± 0.0032 | +0.7 | 0.2291 |
| MIT-BIH lead MLII from lead 2 (mV) | draft_dual | transformer | 3 | 3804 | 0.2234 ± 0.0025 | 0.2204 ± 0.0035 | +1.4 | 0.2303 |
| MIT-BIH lead MLII from lead 2 (mV) | gtmix | mamba | 3 | 4538 | 0.2949 ± 0.0023 | 0.2949 ± 0.0023 | -0.0 | 0.2949 |
| MIT-BIH lead MLII from lead 2 (mV) | gtmix | rpa | 3 | 3530 | 0.3001 ± 0.0130 | 0.2894 ± 0.0143 | +3.7 | 0.2883 |
| MIT-BIH lead MLII from lead 2 (mV) | gtmix | transformer | 3 | 3804 | 0.2965 ± 0.0036 | 0.2952 ± 0.0045 | +0.5 | 0.2960 |
| ETTh1 oil temperature (degC) | draft_bidir | mamba | 3 | 5242 | 4.8504 ± 0.5389 | 4.8046 ± 0.5941 | +1.1 | 5.4984 |
| ETTh1 oil temperature (degC) | draft_bidir | rpa | 3 | 4234 | 5.2647 ± 0.1330 | 5.2968 ± 0.1144 | -0.6 | 7.4196 |
| ETTh1 oil temperature (degC) | draft_bidir | transformer | 3 | 4508 | 4.6715 ± 0.5419 | 4.0717 ± 0.5471 | +15.3 | 5.3818 |
| ETTh1 oil temperature (degC) | draft_causal | mamba | 3 | 5242 | 4.9212 ± 0.4236 | 4.9188 ± 0.4260 | +0.1 | 5.4951 |
| ETTh1 oil temperature (degC) | draft_causal | rpa | 3 | 4234 | 5.1914 ± 0.1742 | 5.3049 ± 0.1323 | -2.1 | 6.7299 |
| ETTh1 oil temperature (degC) | draft_causal | transformer | 3 | 4508 | 4.6152 ± 0.5783 | 4.6681 ± 0.7119 | -0.7 | 6.1943 |
| ETTh1 oil temperature (degC) | draft_dual | mamba | 3 | 5242 | 4.8425 ± 0.5674 | 4.8216 ± 0.5925 | +0.5 | 5.4535 |
| ETTh1 oil temperature (degC) | draft_dual | rpa | 3 | 4234 | 5.3024 ± 0.1694 | 5.3799 ± 0.0910 | -1.5 | 7.2443 |
| ETTh1 oil temperature (degC) | draft_dual | transformer | 3 | 4508 | 4.6990 ± 0.5635 | 4.7849 ± 0.7348 | -1.3 | 6.1799 |
| ETTh1 oil temperature (degC) | gtmix | mamba | 3 | 5242 | 5.5944 ± 0.4521 | 5.6001 ± 0.4455 | -0.1 | 6.7562 |
| ETTh1 oil temperature (degC) | gtmix | rpa | 3 | 4234 | 5.5203 ± 0.3489 | 5.2669 ± 0.3694 | +4.9 | 6.1394 |
| ETTh1 oil temperature (degC) | gtmix | transformer | 3 | 4508 | 6.0285 ± 0.2648 | 5.6913 ± 0.0360 | +6.0 | 7.8574 |
| ETTh1 oil temperature (+OT lag 1) (degC) | draft_dual | rpa | 3 | 4298 | 0.6519 ± 0.0069 | 0.6503 ± 0.0092 | +0.2 | 0.6957 |
| ETTh1 oil temperature (+OT lag 1) (degC) | gtmix | rpa | 3 | 4298 | 0.6562 ± 0.0023 | 0.6390 ± 0.0047 | +2.7 | 0.7101 |
