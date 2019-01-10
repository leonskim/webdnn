'use strict';

const getNormalizedFloat32Array = (imageData) => {
  if (!imageData) {
    return null;
  }
  const means = [0.622, 0.545, 0.520];
  const stds = [0.569, 0.578, 0.585];

  const { width, height } = imageData;

  let float32Array = new Float32Array(width * height * 3);

  // HWC
  /*
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      for (let i = 0; i < 3; i++) {
        let channelVal = imageData.data[(row * width + col) * 4 + i];
        float32Array[(row * width + col) * 3 + i] = (channelVal / 255.0 - means[i]) / stds[i];
      }
    }
  }
  */

  // CHW
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      float32Array[(0 * height + row) * width + col] = (imageData.data[(row * width + col) * 4 + 0] / 255.0 - means[0]) / stds[0];
      float32Array[(1 * height + row) * width + col] = (imageData.data[(row * width + col) * 4 + 1] / 255.0 - means[1]) / stds[1];
      float32Array[(2 * height + row) * width + col] = (imageData.data[(row * width + col) * 4 + 2] / 255.0 - means[2]) / stds[2];
    }
  }
  return float32Array;
}


async function run_entry() {
    try {
        await run();
        log('Run finished');

    } catch (error) {
        log('Error: ' + error);
    }
}

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}


function log(msg) {
    let msg_node = document.getElementById('messages');
    msg_node.appendChild(document.createElement('br'));
    msg_node.appendChild(document.createTextNode(msg));
}

async function loadImage() {
    let imageData = await WebDNN.Image.getImageArray(document.getElementById("image_url").value, {dstW: 178, dstH: 218});
    WebDNN.Image.setImageArrayToCanvas(imageData, 178,218, document.getElementById('input_image'));

    document.getElementById('run_button').disabled = false;
    log('Image loaded to canvas');
}

let runners = {};

function getFrameworkName() {
    return document.querySelector('input[name=framework]:checked').value;
}

async function prepare_run() {
    let backend_name = document.querySelector('input[name=backend]:checked').value;
    let framework_name = getFrameworkName();
    let backend_key = backend_name + framework_name;
    if (!(backend_key in runners)) {
        log('Initializing and loading model');
        let runner = await WebDNN.load(`./output_${framework_name}`, {backendOrder: backend_name});
        log(`Loaded backend: ${runner.backendName}, version: ${runner.descriptor.converted_at}`);

        runners[backend_key] = runner;
    } else {
        log('Model is already loaded');
    }
    return runners[backend_key];
}

async function run() {
    let runner = await prepare_run();
    let x = runner.inputs[0];
    let y = runner.outputs[0];

    let image_options = {
        order: WebDNN.Image.Order.HWC,
        color: WebDNN.Image.Color.BGR,
        bias: [123.68, 116.779, 103.939],
    };

    if (getFrameworkName() === 'chainer' || getFrameworkName() === 'pytorch') {
        image_options.order = WebDNN.Image.Order.CHW;
    }

    if (getFrameworkName() === 'pytorch') {
        image_options.color = WebDNN.Image.Color.RGB;
        
        image_options.scale = [58.40, 57.12, 57.38];
    }


  const canvas = document.getelementbyid('input_image');
  let ctx = canvas.getcontext('2d');
  //ctx.fillstyle = "rgb(200,0,0)";
  //ctx.fillrect(0,0,canvas.width, canvas.height);
  let tempdata = ctx.getimagedata(0, 0, canvas.width, canvas.height);

  x.set(getnormalizedfloat32array(tempdata));

  //x.set(await WebDNN.Image.getImageArray(document.getElementById("input_image"), image_options));

    let start = performance.now();
    await runner.run();
    let elapsed_time = performance.now() - start;
  //console.log('wtf:',y.toActual())
    let top_labels_age1 = softmax(y.slice(0,6));
    // console.log(top_labels_age)
    let top_labels_age=WebDNN.Math.argmax(top_labels_age1);
    let top_labels_eth1 = softmax(y.slice( 6,13));
    let top_labels_eth=WebDNN.Math.argmax(top_labels_eth1);
    let top_labels_hc1 = softmax(y.slice(13,20));
    let top_labels_hc=WebDNN.Math.argmax(top_labels_hc1);
    let top_labels_sv1 = softmax(y.slice(20,25));
    let top_labels_sv=WebDNN.Math.argmax(top_labels_sv1);
    // console.log('Age',Age)
    let predicted_str = 'Predicted:';
    // for (let j = 0; j < top_labels.length; j++) {
    //     predicted_str += ` ${top_labels[j]}(${imagenet_labels[top_labels[j]]})`;
  //     }


  console.log(top_labels_eth1);

    predicted_str=` ${Age[top_labels_age]} (${top_labels_age1[top_labels_age]}) `;
    log('Age')
    log(predicted_str);
    predicted_str=`${Ethnicity[top_labels_eth]} (${top_labels_eth1[top_labels_eth]})`;
    log('Ethnicity')
    log(predicted_str);
    predicted_str=`${Hair_Color[top_labels_hc]} (${top_labels_hc1[top_labels_hc]})`;
    log('Hair_Color')
    log(predicted_str);
    predicted_str=`${Skin_Value[top_labels_sv]} (${top_labels_sv1[top_labels_sv]})`;
    log('Skin Value')
    log(predicted_str);

    console.log('output vector: ', y.toActual());
    log(`Total Elapsed Time[ms/image]: ${elapsed_time.toFixed(2)}`);
}
