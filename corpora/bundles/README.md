`portable.zip` was written with Info-ZIP 3.0 (`zip -X -r -y`) and checked with
`unzip -t`. It is identical to `flatppl-rust/crates/bundle/tests/fixtures/portable.zip`
(SHA-256 `1fa3bc719252664070f4edadb001e2b2dcbdadbea55bf7815dba6e6f59e05c32`).
It contains a Normal prior module, observations `[1, 2]`, a binary
file `[0, 1, 255, 128]`, a relative symlink, and an independent nested archive.
The log density at zero is `-1.5*log(2*pi) - 2.5`, with derivative `3`.
