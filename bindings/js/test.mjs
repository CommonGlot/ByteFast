// node bindings/js/test.mjs MODEL "text" EXPECTED_LABEL
import {loadModel} from "./index.js";

const [path, text, expected] = process.argv.slice(2);
const model = await loadModel(path);
const top = model.predict(text, 3);
console.log(top);
if (top[0].label !== expected || !model.labels.includes(expected) || model.dimension <= 0) {
  throw new Error("unexpected prediction");
}
let failed = false;
try {
  await loadModel(new Uint8Array([1, 2, 3]));
} catch (e) {
  failed = true;
}
if (!failed) {
  throw new Error("invalid model bytes were accepted");
}
model.free();
console.log("JS_PASS");
