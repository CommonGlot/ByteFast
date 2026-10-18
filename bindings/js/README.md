# bytefast (JavaScript)

[ByteFast](https://github.com/CommonGlot/ByteFast) prediction in Node and
browsers, compiled to WebAssembly.

```bash
npm install bytefast
```

```js
import {loadModel} from "bytefast";

const model = await loadModel("model.safetensors"); // a path in Node, a URL, or bytes
console.log(model.predict("The striker scored twice", 3));
// [{label: "__label__sports", probability: 0.98}, ...]
console.log(model.labels, model.dimension);
model.free();
```

Models are trained with the `bytefast` command line or the Python package.
Build from a clone with `make wasm` (needs
[emscripten](https://emscripten.org/)).
