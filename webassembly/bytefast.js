/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import fastTextModularized from './bytefast_wasm.js';
const fastTextModule = fastTextModularized();

let postRunFunc = null;
const addOnPostRun = function(func) {
  postRunFunc = func;
};

fastTextModule.addOnPostRun(() => {
  if (postRunFunc) {
    postRunFunc();
  }
});

const thisModule = this;
const trainFileInWasmFs = 'train.txt';
const testFileInWasmFs = 'test.txt';
const modelFileInWasmFs = 'model.bin';

const getFloat32ArrayFromHeap = (len) => {
  const dataBytes = len * Float32Array.BYTES_PER_ELEMENT;
  const dataPtr = fastTextModule._malloc(dataBytes);
  const dataHeap = new Uint8Array(fastTextModule.HEAPU8.buffer,
    dataPtr,
    dataBytes);
  return {
    'ptr':dataHeap.byteOffset,
    'size':len,
    'buffer':dataHeap.buffer
  };
};

const heapToFloat32 = (r) => new Float32Array(r.buffer, r.ptr, r.size);

class ByteFast {
  constructor() {
    this.f = new fastTextModule.ByteFast();
  }

  /**
   * loadModel
   *
   * Loads the model file from the specified url, and returns the
   * corresponding `FastTextModel` object.
   *
   * @param {string}     url
   *     the url of the model file.
   *
   * @return {Promise}   promise object that resolves to a `FastTextModel`
   *
   */
  loadModel(url) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;

    const fastTextNative = this.f;
    return new Promise(function(resolve, reject) {
      fetchFunc(url).then(response => {
        return response.arrayBuffer();
      }).then(bytes => {
        const byteArray = new Uint8Array(bytes);
        const FS = fastTextModule.FS;
        FS.writeFile(modelFileInWasmFs, byteArray);
      }).then(() =>  {
        fastTextNative.loadModel(modelFileInWasmFs);
        resolve(new FastTextModel(fastTextNative));
      }).catch(error => {
        reject(error);
      });
    });
  }

  _train(url, modelName, kwargs = {}, callback = null) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;

    return new Promise(function(resolve, reject) {
      fetchFunc(url).then(response => {
        return response.arrayBuffer();
      }).then(bytes => {
        const byteArray = new Uint8Array(bytes);
        const FS = fastTextModule.FS;
        FS.writeFile(trainFileInWasmFs, byteArray);
      }).then(() =>  {
        const argsList = ['lr', 'lrUpdateRate', 'dim', 'epoch',
          'minCount', 'minCountLabel', 'neg', 'wordNgrams', 'bucket',
          'minn', 'maxn', 'minb', 'maxb', 'minc', 'maxc', 'maxBytes',
          'hashWords', 'wordlist', 'maxVocab', 'cropProb', 'cropMin',
          'cropMax', 'cropUnit', 'normProb', 'label', 'verbose', 'seed'];
        const args = new fastTextModule.Args();
        argsList.forEach(k => {
          if (k in kwargs) {
            args[k] = kwargs[k];
          }
        });
        args.loss = fastTextModule.LossName[kwargs['loss'] || 'softmax'];
        args.thread = 1;
        args.input = trainFileInWasmFs;

        fastTextNative.train(args, callback);

        resolve(new FastTextModel(fastTextNative));
      }).catch(error => {
        reject(error);
      });
    });
  }

  /**
   * trainSupervised
   *
   * Downloads the input file from the specified url, trains a supervised
   * model and returns a `FastTextModel` object.
   *
   * @param {string}     url
   *     the url of the input file.
   *     The input file must must contain at least one label per line. For an
   *     example consult the example datasets which are part of the fastText
   *     repository such as the dataset pulled by classification-example.sh.
   *
   * @param {dict}       kwargs
   *     train parameters.
   *     For example {'lr': 0.5, 'epoch': 5}
   *
   * @param {function}   callback
   *     train callback function
   *     `callback` function is called regularly from the train loop:
   *     `callback(progress, loss, wordsPerSec, learningRate, eta)`
   *
   * @return {Promise}   promise object that resolves to a `FastTextModel`
   *
   */
  trainSupervised(url, kwargs = {}, callback) {
    const self = this;
    return new Promise(function(resolve, reject) {
      self._train(url, 'supervised', kwargs, callback).then(model => {
        resolve(model);
      }).catch(error => {
        reject(error);
      });
    });
  }

}


class FastTextModel {
  /**
     * `FastTextModel` represents a trained model.
     *
     * @constructor
     *
     * @param {object}       fastTextNative
     *     webassembly object that makes the bridge between js and C++
     */
  constructor(fastTextNative) {
    this.f = fastTextNative;
  }

  /**
     * isQuant
     *
     * @return {bool}   true if the model is quantized
     *
     */
  isQuant() {
    return this.f.isQuant;
  }

  /**
     * getDimension
     *
     * @return {int}    the dimension (size) of a lookup vector (hidden layer)
     *
     */
  getDimension() {
    return this.f.args.dim;
  }

  /**
     * predict
     *
     * Given a string, get a list of labels and a list of corresponding
     * probabilities. k controls the number of returned labels.
     *
     * @param {string}          text
     * @param {int}             k, the number of predictions to be returned
     * @param {number}          probability threshold
     *
     * @return {Array.<Pair.<number, string>>}
     *     labels and their probabilities
     *
     */
  predict(text, k = 1, threshold = 0.0) {
    return this.f.predict(text, k, threshold);
  }

  /**
     * getWords
     *
     * Get the entire list of words of the dictionary including the frequency
     * of the individual words. This does not include any subwords. For that
     * please consult the function get_subwords.
     *
     * @return {Pair.<Array.<string>, Array.<int>>}
     *     words and their corresponding frequencies
     *
     */
  getWords() {
    return this.f.getWords();
  }

  /**
     * getLabels
     *
     * Get the entire list of labels of the dictionary including the frequency
     * of the individual labels.
     *
     * @return {Pair.<Array.<string>, Array.<int>>}
     *     labels and their corresponding frequencies
     *
     */
  getLabels() {
    return this.f.getLabels();
  }

  /**
     * getLine
     *
     * Split a line of text into words and labels. Labels must start with
     * the prefix used to create the model (__label__ by default).
     *
     * @param {string}          text
     *
     * @return {Pair.<Array.<string>, Array.<string>>}
     *     words and labels
     *
     */
  getLine(text) {
    return this.f.getLine(text);
  }

  /**
     * saveModel
     *
     * Saves the model file in web assembly in-memory FS and returns a blob
     *
     * @return {Blob}           blob data of the file saved in web assembly FS
     *
     */
  saveModel() {
    this.f.saveModel(modelFileInWasmFs);
    const content = fastTextModule.FS.readFile(modelFileInWasmFs,
      { encoding: 'binary' });
    return new Blob(
      [new Uint8Array(content, content.byteOffset, content.length)],
      { type: ' application/octet-stream' }
    );
  }

  /**
     * test
     *
     * Downloads the test file from the specified url, evaluates the supervised
     * model with it.
     *
     * @param {string}          url
     * @param {int}             k, the number of predictions to be returned
     * @param {number}          probability threshold
     *
     * @return {Promise}   promise object that resolves to a `Meter` object
     *
     * example:
     * model.test("/absolute/url/to/test.txt", 1, 0.0).then((meter) => {
     *     console.log(meter.precision);
     *     console.log(meter.recall);
     *     console.log(meter.f1Score);
     *     console.log(meter.nexamples());
     * });
     *
     */
  test(url, k, threshold) {
    const fetchFunc = (thisModule && thisModule.fetch) || fetch;
    const fastTextNative = this.f;

    return new Promise(function(resolve, reject) {
      fetchFunc(url).then(response => {
        return response.arrayBuffer();
      }).then(bytes => {
        const byteArray = new Uint8Array(bytes);
        const FS = fastTextModule.FS;
        FS.writeFile(testFileInWasmFs, byteArray);
      }).then(() =>  {
        const meter = fastTextNative.test(testFileInWasmFs, k, threshold);
        resolve(meter);
      }).catch(error => {
        reject(error);
      });
    });
  }
}


export {ByteFast, addOnPostRun};
