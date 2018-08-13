/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import {Array2D, Graph, NDArray, NDArrayMathGPU, Session, SGDOptimizer} from
// 'deeplearn'; import {InCPUMemoryShuffledInputProviderBuilder} from
// 'deeplearn/dist/data/input_provider'; import {Tensor} from
// 'deeplearn/dist/graph/graph'; import {AdagradOptimizer} from
// 'deeplearn/dist/graph/optimizers/adagrad_optimizer'; import {AdamaxOptimizer}
// from 'deeplearn/dist/graph/optimizers/adamax_optimizer';
// import {CostReduction, FeedEntry} from 'deeplearn/dist/graph/session';
// import {NDArrayMath} from 'deeplearn/dist/math/math';
// import {Scalar} from 'deeplearn/dist/math/ndarray';
// import {expectArrayInMeanStdRange} from 'deeplearn/dist/test_util';
// import {Server} from 'http';
// import {setTimeout} from 'timers';
var tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
// import {Tensor, Tensor2D} from '@tensorflow/tfjs';
// import * as tf from '@tensorflow/tfjs-node';

/** TODO(kreeger): Doc me. */
function GameOfLife(size) {
  // math: NDArrayMath;
  this.size = size;

}

GameOfLife.prototype.setSize = function(size) {
    this.size = size;
};



  /* Helper method to pad an array until the op is ready. */
  var padArray=function(array) {
    const x1 = array.shape[0];
    const x2 = array.shape[1];
    const pad = 1;

    const oldValues = array.dataSync();
    const shape = [x1 + pad * 2, x2 + pad * 2];
    const values = [];

    let z = 0;
    for (let i = 0; i < shape[0]; i++) {
      let rangeStart = -1;
      let rangeEnd = -1;
      if (i > 0 && i < shape[0] - 1) {
        rangeStart = i * shape[1] + 1;
        rangeEnd = i * shape[1] + x2;
      }
      for (let j = 0; j < shape[1]; j++) {
        const v = i * shape[0] + j;
        if (v >= rangeStart && v <= rangeEnd) {
          values[v] = oldValues[z++];
        } else {
          values[v] = 0;
        }
      }
    }
    return tf.tensor2d(values, shape);
  };



  /** Counts total sum of neighbors for a given world. */
  var countNeighbors = function(size, worldPadded) {
    let neighborCount = tf.add(
        tf.slice(worldPadded, [0, 0], [size - 2, size - 2]),
        tf.slice(worldPadded, [0, 1], [size - 2, size - 2]));
    neighborCount = tf.add(
        neighborCount, tf.slice(worldPadded, [0, 2], [size - 2, size - 2]));
    neighborCount = tf.add(
        neighborCount, tf.slice(worldPadded, [1, 0], [size - 2, size - 2]));
    neighborCount = tf.add(
        neighborCount, tf.slice(worldPadded, [1, 2], [size - 2, size - 2]));
    neighborCount = tf.add(
        neighborCount, tf.slice(worldPadded, [2, 0], [size - 2, size - 2]));
    neighborCount = tf.add(
        neighborCount, tf.slice(worldPadded, [2, 1], [size - 2, size - 2]));
    neighborCount = tf.add(
        neighborCount, tf.slice(worldPadded, [2, 2], [size - 2, size - 2]));
    return neighborCount;
  };

  GameOfLife.prototype.generateGolExample = function() {
    const world =
        tf.randomUniform([this.size - 2, this.size - 2], 0, 2, 'int32');

    const worldPadded = padArray(world);
    const numNeighbors = countNeighbors(this.size, worldPadded).dataSync();
    const worldValues = world.dataSync();
    const nextWorldValues = [];
    for (let i = 0; i < numNeighbors.length; i++) {
      const value = numNeighbors[i];
      let nextVal = 0;
      if (value === 3) {
        // Cell rebirths
        nextVal = 1;
      } else if (value === 2) {
        // Cell survives
        nextVal = worldValues[i];
      } else {
        // Cell dies
        nextVal = 0;
      }
      nextWorldValues.push(nextVal);
    }
    const worldNext = tf.tensor2d(nextWorldValues, world.shape, 'int32');
    return [worldPadded, padArray(worldNext)];
  };


/* Draws Game Of Life sequences */
// class WorldDisplay {
//   rootElement: Element;

//   constructor() {
//     this.rootElement = document.createElement('div');
//     this.rootElement.setAttribute('class', 'world-display');

//     document.querySelector('.worlds-display').appendChild(this.rootElement);
//   }

//   displayWorld(world: Tensor2D, title: string): Element {
//     let worldElement = document.createElement('div');
//     worldElement.setAttribute('class', 'world');

//     let titleElement = document.createElement('div');
//     titleElement.setAttribute('class', 'title');
//     titleElement.innerText = title;
//     worldElement.appendChild(titleElement);

//     let boardElement = document.createElement('div');
//     boardElement.setAttribute('class', 'board');

//     for (let i = 0; i < world.shape[0]; i++) {
//       let rowElement = document.createElement('div');
//       rowElement.setAttribute('class', 'row');

//       for (let j = 0; j < world.shape[1]; j++) {
//         let columnElement = document.createElement('div');
//         columnElement.setAttribute('class', 'column');
//         if (world.get(i, j) == 1) {
//           columnElement.classList.add('alive');
//         } else {
//           columnElement.classList.add('dead');
//         }
//         rowElement.appendChild(columnElement);
//       }
//       boardElement.appendChild(rowElement);
//     }

//     worldElement.appendChild(boardElement);
//     this.rootElement.appendChild(worldElement);
//     return worldElement;
//   }
// }

// class WorldContext {
//   worldDisplay: WorldDisplay;
//   world: Tensor2D;
//   worldNext: Tensor2D;
//   predictionElement: Element = null;

//   constructor(worlds: [Tensor2D, Tensor2D]) {
//     this.worldDisplay = new WorldDisplay();

//     this.world = worlds[0];
//     this.worldNext = worlds[1];
//     this.worldDisplay.displayWorld(this.world, 'Sequence');
//     this.worldDisplay.displayWorld(this.worldNext, 'Next Sequence');
//   }

//   displayPrediction(prediction: Tensor2D) {
//     if (this.predictionElement) {
//       this.predictionElement.remove();
//     }
//     this.predictionElement =
//         this.worldDisplay.displayWorld(prediction, 'Prediction');
//   }
// }

// class TrainDisplay {
//   element: Element;

//   constructor() {
//     this.element = document.querySelector('.train-display');
//   }

//   logCost(cost: number) {
//     const costElement = document.createElement('div');
//     costElement.setAttribute('class', 'cost');
//     costElement.innerText = '* Cost: ' + cost;
//     this.element.appendChild(costElement);
//   }

//   showStep(step: number, steps: number) {
//     this.element.innerHTML = 'Trained ' + Math.trunc(step / steps * 100) +
//     '%';
//   }
// }

function reshape(inputArray, rows, cols) {
  var outputArray = [];

  for (var r = 0; r < rows; r++) {
    var row = [];
    for (var c = 0; c < cols; c++) {
      var i = r * cols + c;
      if (i < inputArray.length) {
        row.push(inputArray[i]);
      }
    }
    outputArray.push(row);
  }
  return outputArray;
}

function getModel(game) {
  const initialLearningRate = 0.042;
  const model = tf.sequential();
  // model.add(
  //     tf.layers.dense({inputShape: [5, 5], activation: 'relu', units: 25}));
  //     model.add(
  //         tf.layers.dense({inputShape: [5, 5], activation: 'relu', units: 25}));
  // model.add(tf.layers.dense({activation: 'relu', units: 25}));
  // model.add(tf.layers.dense({activation: 'sigmoid', units: 25}));
  // model.add(tf.layers.dense({activation: 'relu', units: 25}));
  // model.add(tf.layers.dense({activation: 'sigmoid', units: 5}));
  // model.compile({
  //   optimizer: tf.train.adam(),
  //   loss: 'categoricalCrossentropy',
  //   metrics: ['accuracy']
  // });
  model.add(
      tf.layers.dense({inputShape: [25], activation: 'relu', units: 200, useBias:true}));
  model.add(tf.layers.dense({activation: 'relu', units: 200, useBias:true}));
  model.add(tf.layers.dense({activation: 'relu', units: 200, useBias:true}));
  model.add(tf.layers.dense({activation: 'relu', units: 25, useBias:true}));
  model.compile({
    optimizer: tf.train.sgd(initialLearningRate), //tf.train.adam(),
    loss: 'meanSquaredError', //'categoricalCrossentropy',
    metrics: ['accuracy']
  });

// model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});



  return model;
}

    // Setup game
    // const math = new NDArrayMathGPU();
    const game = new GameOfLife(5 /*, math*/);
    const model = getModel(game /*, math*/);
    const batchSize = 10000;

    const inputs = [];
    const outputs = [];
    for (let i = 0; i < batchSize; i++) {
      const example = game.generateGolExample();
      inputs.push(Array.from(example[0].dataSync()));
      outputs.push(Array.from(example[1].dataSync()));
      // inputs.push(reshape(Array.from(example[0].dataSync()), 5, 5));
      // outputs.push(reshape(Array.from(example[1].dataSync()), 5, 5));
      // model.fit(
      //     example[0].reshape([game.size * game.size]),
      //     example[1].reshape([game.size * game.size]), {epochs: 10});
    }
    const trainData = tf.tensor(inputs, [batchSize, 25]);
    const trainoutputsData = tf.tensor(outputs, [batchSize, 25]);
    model.fit(trainData, trainoutputsData, {batchSize:batchSize, epochs: 25}).then((history) => {
      console.log(history);
      model.save('file:///usr/local/google/home/kangyizhang/test/temp-model');
      const example = game.generateGolExample();
      example[0].print();
      example[1].print();
      const result = model
          .predict(tf.tensor([Array.from(example[0].dataSync())], [1, 25]));
          // result.print();
          // .reshape([game.size, game.size])
      console.log(reshape(result.dataSync(), 5, 5));
    });
// // Helper classes for displaying worlds and training data:
// const trainDisplay = new TrainDisplay();
// const worldDisplay = new WorldDisplay();

// // List of worlds + display contexts.
// let worldContexts: Array<WorldContext> = [];

// const boardSizeInput =
//     document.getElementById('board-size-input') as HTMLTextAreaElement;
// const trainingSizeInput =
//     document.getElementById('training-size-input') as HTMLTextAreaElement;
// const learningRateInput =
//     document.getElementById('learning-rate-input') as HTMLTextAreaElement;
// const numLayersInput =
//     document.getElementById('num-layers-input') as HTMLTextAreaElement;
// const addSequenceButton = document.querySelector('.add-sequence-button');
// const trainButton = document.querySelector('.train-button');
// const predictButton = document.querySelector('.predict-button');
// const resetButton = document.querySelector('.reset-button');

// function getBoardSize() {
//   return parseInt(boardSizeInput.value);
// }

// function clearChildNodes(node: Element) {
//   while (node.hasChildNodes()) {
//     node.removeChild(node.lastChild);
//   }
// }

// let step = 0;
// let trainLength = 0;
// function trainAndRender() {
//   if (step == trainLength) {
//     trainButton.removeAttribute('disabled');
//     predictButton.removeAttribute('disabled');
//     resetButton.removeAttribute('disabled');
//     boardSizeInput.removeAttribute('disabled');
//     learningRateInput.removeAttribute('disabled');
//     trainingSizeInput.removeAttribute('disabled');
//     numLayersInput.removeAttribute('disabled');
//     return;
//   }

//   requestAnimationFrame(trainAndRender);
//   step++;

//   const fetchCost = step % 10 == 0;
//   const cost = model.trainBatch(fetchCost);

//   if (fetchCost) {
//     trainDisplay.showStep(step, trainLength);
//   }
// }

// addSequenceButton.addEventListener('click', () => {
//   game.setSize(getBoardSize());
//   worldContexts.push(new WorldContext(game.generateGolExample()));
// });

// trainButton.addEventListener('click', () => {
//   trainButton.setAttribute('disabled', 'disabled');
//   predictButton.setAttribute('disabled', 'disabled');
//   resetButton.setAttribute('disabled', 'disabled');
//   boardSizeInput.setAttribute('disabled', 'disabled');
//   learningRateInput.setAttribute('disabled', 'disabled');
//   trainingSizeInput.setAttribute('disabled', 'disabled');
//   numLayersInput.setAttribute('disabled', 'disabled');

//   const boardSize = getBoardSize();
//   const learningRate = parseFloat(learningRateInput.value);
//   const trainingSize = parseInt(trainingSizeInput.value);
//   const numLayers = parseInt(numLayersInput.value);

//   game.setSize(boardSize);
//   model.setupSession(boardSize, learningRate, numLayers);

//   step = 0;
//   trainLength = trainingSize;
//   trainAndRender();
// });

// predictButton.addEventListener('click', () => {
//   worldContexts.forEach((worldContext) => {
//     worldContext.displayPrediction(model.predict(worldContext.world));
//   });
// });

// resetButton.addEventListener('click', () => {
//   worldContexts = [];
//   clearChildNodes(document.querySelector('.worlds-display'));
//   clearChildNodes(document.querySelector('.train-display'));
// });



/**
 * Main class for running a deep-neural network of training for Game-of-life
 * next sequence.
 */
// class GameOfLifeModel {
//   // session: Session;
//   // math: NDArrayMath;
//   batchSize = 1;

//   // An optimizer with a certain initial learning rate. Used for training.
//   initialLearningRate = 0.042;
//   optimizer: tf.SGDOptimizer;
//   // optimizer: tf.AdagradOptimizer;

//   inputTensor: Tensor;
//   targetTensor: Tensor;
//   costTensor: Tensor;
//   predictionTensor: Tensor;

//   size: number;
//   step = 0;

//   // Maps tensors to InputProviders
//   feedEntries: FeedEntry[];

//   game: GameOfLife;

//   constructor(game: GameOfLife /*, math: NDArrayMath*/) {
//     // this.math = math;
//     this.game = game;
//   }

//   setupSession(
//       boardSize: number, initialLearningRate: number, numLayers: number):
//       void {
//     this.optimizer = new SGDOptimizer(this.initialLearningRate);

//     this.size = boardSize;
//     const graph = new Graph();
//     const shape = this.size * this.size;

//     this.inputTensor = graph.placeholder('input', [shape]);
//     this.targetTensor = graph.placeholder('target', [shape]);

//     let hiddenLayer = GameOfLifeModel.createFullyConnectedLayer(
//         graph, this.inputTensor, 0, shape);
//     for (let i = 1; i < numLayers; i++) {
//       hiddenLayer = GameOfLifeModel.createFullyConnectedLayer(
//           graph, hiddenLayer, i, shape);
//     }

//     this.predictionTensor = hiddenLayer;

//     this.costTensor =
//         graph.meanSquaredCost(this.targetTensor, this.predictionTensor);
//     this.session = new Session(graph, this.math);
//   }

//   trainBatch(shouldFetchCost: boolean): number {
//     this.generateTrainingData();
//     // Every 42 steps, lower the learning rate by 15%.
//     const learningRate = this.initialLearningRate *
//         Math.pow(0.85, Math.floor(this.step++ / 100));
//     this.optimizer.setLearningRate(learningRate);
//     let costValue = -1;
//     this.math.scope(() => {
//       const cost = this.session.train(
//           this.costTensor, this.feedEntries, this.batchSize,
//           this.optimizer, shouldFetchCost ? CostReduction.MEAN :
//           CostReduction.NONE);

//       if (!shouldFetchCost) {
//         return;
//       }
//       costValue = cost.get();
//     });
//     return costValue;
//   }

//   predict(world: Tensor2D): Tensor2D {
//     let values = null;
//     this.math.scope((keep, track) => {
//       const mapping = [{
//         tensor: this.inputTensor,
//         data: world.reshape([this.size * this.size])
//       }]

//           const evalOutput = this.session.eval(this.predictionTensor,
//           mapping);
//       values = evalOutput.getValues();
//     });
//     return tf.tensor2d(values, [this.size, this.size]);
//   }

//   private generateTrainingData(): void {
//     // this.math.scope(() => {
//     const inputs = [];
//     const outputs = [];
//     for (let i = 0; i < this.batchSize; i++) {
//       const example = this.game.generateGolExample();
//       inputs.push(example[0].reshape([this.size * this.size]));
//       outputs.push(example[1].reshape([this.size * this.size]));
//     }

//     // TODO(kreeger): Don't really need to shuffle these.
//     // const inputProviderBuilder =
//     //     new InCPUMemoryShuffledInputProviderBuilder([inputs, outputs]);
//     const [inputProvider, targetProvider] =
//         inputProviderBuilder.getInputProviders();

//     this.feedEntries = [
//       {tensor: this.inputTensor, data: inputProvider},
//       {tensor: this.targetTensor, data: targetProvider}
//     ];
//     // });
//   }

//   /* Helper method for creating a fully connected layer. */
//   private static createFullyConnectedLayer(
//       graph: Graph, inputLayer: Tensor, layerIndex: number,
//       sizeOfThisLayer: number, includeRelu = true, includeBias = true):
//       Tensor {
//     return graph.layers.dense(
//         'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
//         includeRelu ? (x) => graph.relu(x) : undefined, includeBias);
//   }
// }
