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
var GameOfLife = require('./src/game');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var exphbs  = require('express-handlebars');
var tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

var model;

async function loadModel() {
  try {
  const game = new GameOfLife(5);

  model = await tf.loadModel('file:///usr/local/google/home/kangyizhang/test/temp-model/model.json');

  const example = game.generateGolExample();
    example[0].print();
    example[1].print();


    const result =model.predict(tf.tensor([Array.from(example[0].dataSync())], [1, 25]));

    console.log(result.reshape([5, 5]).print());
  } catch(err) {
    console.log(err);
  }
}

loadModel();

var app = express();

// view engine setup
app.set('views', __dirname);
app.set('view engine', 'jade');

app.get('/', (req, res) => res.render('index'));

app.listen(3000, () => console.log('Example app listening on port 3000!'));
