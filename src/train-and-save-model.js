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
const GameOfLife = require('./game');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');


function getModel(game) {
  const initialLearningRate = 0.042;
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {inputShape: [25], activation: 'relu', units: 25, useBias: true}));
  model.add(tf.layers.dense({activation: 'relu', units: 25, useBias: true}));
  model.add(tf.layers.dense({activation: 'relu', units: 25, useBias: true}));
  model.add(tf.layers.dense({activation: 'relu', units: 25, useBias: true}));
  model.compile({
    optimizer: tf.train.sgd(initialLearningRate),  // tf.train.adam(),
    loss: 'meanSquaredError',                      //'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}


async function trainAndSaveModel() {
  // Setup game
  const game = new GameOfLife(5 /*, math*/);
  const model = getModel(game /*, math*/);
  const batchSize = 10000;

  const inputs = [];
  const outputs = [];
  for (let i = 0; i < batchSize; i++) {
    const example = game.generateGolExample();
    inputs.push(Array.from(example[0].dataSync()));
    outputs.push(Array.from(example[1].dataSync()));
  }
  const trainData = tf.tensor(inputs, [batchSize, 25]);
  const trainoutputsData = tf.tensor(outputs, [batchSize, 25]);
  model.fit(trainData, trainoutputsData, {batchSize: batchSize, epochs: 25})
      .then((history) => {
        console.log(history);
        const example = game.generateGolExample();
        example[0].print();
        example[1].print();
        const result = model.predict(
            tf.tensor([Array.from(example[0].dataSync())], [1, 25]));
        console.log(result.reshape([5, 5]).print());
      });

  await model.save('file:///usr/local/google/home/kangyizhang/test/temp-model');
}

trainAndSaveModel();
