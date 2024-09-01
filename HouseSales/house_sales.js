async function predict () {
    alert("Not yet implemented");
}

async function load () {
    if (doesModelExist()) {
        const models = await tf.io.listModels();
        const modelInfo = models[storageKey];
        model = await tf.loadLayersModel(storageKey);
        showModelSummary(model);
        document.getElementById("model-status").innerHTML = `Trained (load model from saved ${modelInfo.dateSaved})`;
        document.getElementById("test-button").removeAttribute("disabled");
        document.getElementById("predict-button").removeAttribute("disabled");
    } else {
        alert('Could not load: no saved model found');
    }   
}

async function doesModelExist () {    
    const models = await tf.io.listModels();
    const modelInfo = models[storageKey];
    return modelInfo? true : false;    
}

const storageID = "kc-house-price-regresssion";
const storageKey = `localstorage://${storageID}`;
async function save () {
    const saveResults = await model.save(`localstorage://${storageID}`);
    document.getElementById("model-status").innerHTML = `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`;
    document.getElementById("load-button").removeAttribute("disabled");
}

async function test () { 
    compileModel(model);           
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = await lossTensor.dataSync();
    console.log(`Testing set loss: ${loss}`);

    document.getElementById("testing-status").innerHTML = `Testing set loss: ${loss[0].toPrecision(5)}`;
}

async function train() {
    // disable all button and update the model status
    ["train","test","load","predict","save"].forEach(id => {
        document.getElementById(`${id}-button`).setAttribute("disabled", "disabled");
    });
    document.getElementById("model-status").innerHTML = "Training ...";

    const model = createModel();
    //model.summary(); // console log
    showModelSummary(model);

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    console.log(result);
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);
    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);

    document.getElementById("model-status").innerHTML = `Trained (unsaved)\n`
        + `Loss: ${trainingLoss.toPrecision(5)}\n`
        + `Validation loss: ${validationLoss.toPrecision(5)}`;
    document.getElementById("test-button").removeAttribute("disabled");
    document.getElementById("save-button").removeAttribute("disabled");
    if (doesModelExist()) {
        document.getElementById("load-button").removeAttribute("disabled");    
    }
}

function showModelSummary(model) {
    tfvis.show.modelSummary({ name: "Model Summary", tab: "Model" }, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1", tab: "Model Inspection" }, layer);
}

async function toggleVisor () {
    tfvis.visor().toggle();
}


async function plot(pointsArray, featureName) {
    const container = { name: `${featureName} vs House Price`, tab: 'Visor' };
    const data = { values: [pointsArray], series: ["original"] };
    const opts = { xLabel: featureName, yLabel: "Price" };
    tfvis.render.scatterplot(container, data, opts);
}

function normalize(tensor) {
    const min = tensor.min();
    const max = tensor.max();
    const normalizedTensor = tensor.sub(min).div(max.sub(min));
    return {
        tensor: normalizedTensor,
        min,
        max
    };
}

function denormalize(tensor, min, max) {
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
    return denormalizedTensor;
}

let model;
function createModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'linear',
        inputDim: 1
    }));

    compileModel(model);

    return model;
}

function compileModel(model) {
    const optimizer = tf.train.sgd(0.1);
    model.compile({
        loss: 'meanSquaredError',
        optimizer
    });
}

async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {
    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
        { name: "Training Performance", tab: "Training" },
        ['loss']
    )
    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32, // 512,
        epochs: 20,
        validationSplit: 0.2,
        shuffle: true,
        callbacks: {
            //onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
            onEpochEnd //, onBatchEnd
        }
    });
}

let normalizedFeature, normalizedLabel;
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;
async function run() {
    await tf.ready();

    // Import from csv
    const houseSalesDataset = tf.data.csv("https://ijianhuang.github.io/HouseSales/kc_house_data.csv")

    // Extract x and y from data set to plot
    const pointsDataset = houseSalesDataset.map(record => ({
        x: record.sqft_living,
        y: record.price
    }));
    const points = await pointsDataset.toArray();
    if (points.length % 2 !== 0) {
        points.pop();
    }
    tf.util.shuffle(points); // attn: in-place shuffle 
    plot(points, "Square Feet");

    // features (inputs)
    const featureValues = await points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    // labels (outputs)
    const labelValues = await points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // normalize features and labels
    normalizedFeature = normalize(featureTensor);
    normalizedLabel = normalize(labelTensor);

    featureTensor.dispose();
    labelTensor.dispose();

    [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2);
    [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2);

    // update status and enable train button
    document.getElementById("model-status").innerHTML = "No model trained";
    document.getElementById("train-button").removeAttribute("disabled");            
    document.getElementById("load-button").removeAttribute("disabled");            
}

run();