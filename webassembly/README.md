# WebAssembly module

Build with [emscripten](https://emscripten.org/) (activate it with
`source /path/to/emsdk/emsdk_env.sh`):

```bash
make wasm        # writes webassembly/bytefast_wasm.js and .wasm
```

`webassembly/bytefast.js` wraps the module:

```js
import {ByteFast, addOnPostRun} from "./bytefast.js";

addOnPostRun(async () => {
  const bf = new ByteFast();
  const model = await bf.loadModel("model.bin");
  console.log(model.predict("The striker scored twice", 3, 0.0));
  console.log(model.getLabels());
});
```

`trainSupervised(url, {lr: 0.5, epoch: 5, minb: 2, maxb: 5}, callback)`
trains in the browser (one thread). The model object also has `getWords`,
`getLine`, `test(url, k, threshold)`, `saveModel`, `isQuant` and
`getDimension`. Serve the page over HTTP (`python3 -m http.server`); see
`doc/examples/` for complete pages.
