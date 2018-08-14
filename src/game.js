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
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

function GameOfLife(size) {
  this.size = size;
}

GameOfLife.prototype.setSize = function(size) {
  this.size = size;
};

/* Helper method to pad an array until the op is ready. */
var padArray = function(array) {
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
  const world = tf.randomUniform([this.size - 2, this.size - 2], 0, 2, 'int32');

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
  // return [world, worldNext];
};

module.exports = GameOfLife;
