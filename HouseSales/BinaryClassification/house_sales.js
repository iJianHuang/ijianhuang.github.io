async function predict () {
    const predictionInput1 = parseInt(document.getElementById("prediction-input-1").value);
    const predictionInput2 = parseInt(document.getElementById("prediction-input-2").value);
    if (isNaN(predictionInput1) || isNaN(predictionInput2)) {
        alert("Please enter a valid number");
    } else if (predictionInput1 < 200 ) {
        alert("Please enter a valid and reasonable house square feet above 200 sqft");
    } else if (predictionInput2 < 75000 ) {
        alert("Please enter a valid and reasonable house price above $75,000");
    }
    else {
        tf.tidy(() => {
            const inputTensor = tf.tensor2d([[predictionInput1, predictionInput2]]);
            const normalizedInput = normalize(inputTensor, normalizedFeature.min, normalizedFeature.max);
            const normalizedOutputTensor = model.predict(normalizedInput.tensor);
            const outputTensor = denormalize(normalizedOutputTensor, normalizedLabel.min, normalizedLabel.max);
            const outputValue = outputTensor.dataSync()[0];
            const outputValueRounded = (outputValue*100).toFixed(1);
            document.getElementById("prediction-output").innerHTML = `The likelihood of being waterfront property is <br>`
                + `<span style="font-size:2em;">\$${outputValueRounded}%</span> `;
        });
    }
}

async function load () {
    if (doesModelExist()) {
        const models = await tf.io.listModels();
        const modelInfo = models[storageKey];
        model = await tf.loadLayersModel(storageKey);
        showModelSummary(model);

        //await plotPredictionLine();
        await plotPredictionHeadmap();
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

const storageID = "kc-house-price-binary";
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
    //await plotPredictionLine();

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    await plotPredictionHeadmap();
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
    document.getElementById("predict-button").removeAttribute("disabled");
    if (doesModelExist()) {
        document.getElementById("load-button").removeAttribute("disabled");    
    }
}

function showModelSummary(model) {
    tfvis.show.modelSummary({ name: "Model Summary", tab: "Visor" }, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1", tab: "Visor" }, layer);
}

async function plotParams(weight, bias) {
    model.getLayer(null, 0).setWeights([
        tf.tensor2d([[weight]]),
        tf.tensor1d([bias])
    ]);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1", tab: "Visor" }, layer);
    //await plotPredictionLine();
}

async function toggleVisor () {
    tfvis.visor().toggle();
}

async function plotClasses(pointsArray, classKey, size = 800, equalizeClassSizes = true) {
    const allSeries = {};
    pointsArray.forEach(p => {
        const seriesName = `${classKey}: ${p.class}`;
        let series = allSeries[seriesName];
        if (!series) {
            series = [];
            allSeries[seriesName] = series;            
        }
        series.push(p);
    });
    
    if (equalizeClassSizes) {
        let maxLength = null;
        Object.values(allSeries).forEach(series => {
            if (maxLength === null
                || (series.length < maxLength && series.length >= 100)) {
                maxLength = series.length;
            }
        });
        Object.keys(allSeries).forEach(keyName => {
            allSeries[keyName] = allSeries[keyName].slice(0, maxLength);
            if (allSeries[keyName].length < 100) {
                delete allSeries[keyName];
            }
        });
    }

    tfvis.render.scatterplot(
        {
            name: "Square Feet vs House Price",
            styles: { width: "100%" }
        },
        {
            values: Object.values(allSeries),
            series: Object.keys(allSeries)
        },
        {
            xLabel: "Square Feet",
            yLabel: "Price",
            height: size,
            width: size * 1.5
        }
    );    
}

async function plot(pointsArray, featureName, predictedPointsArray = null, tab = "Visor") {
    const values = [pointsArray.slice(0, 1000)];
    const series = ["original"];
    if (Array.isArray(predictedPointsArray)) {
        values.push(predictedPointsArray);
        series.push("predicted");
    }
    const container = { name: `${featureName} vs House Price`, tab: tab };
    const data = { values, series };
    const opts = { xLabel: featureName, yLabel: "Price" };
    tfvis.render.scatterplot(container, data, opts);
}

async function plotPredictionLine() {
    const [xs, ys] = tf.tidy(() => {
        const normalizedXs = tf.linspace(0, 1, 100);
        const normalizedYs = model.predict(normalizedXs.reshape([100, 1]));

        const xs = denormalize(normalizedXs, normalizedFeature.min, normalizedFeature.max);
        const ys = denormalize(normalizedYs, normalizedLabel.min, normalizedLabel.max);

        return [xs.dataSync(), ys.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, index) => {
        return { x: val, y: ys[index]};
    });

    await plot(points, "Square feet", predictedPoints, "Training");
}

async function plotPredictionHeadmap(name = "Predicted Class", size = 400) {
    const [valuePromise, xTicksPromise, yTicksPromise] = tf.tidy(() => {
        const gridSize = 50;
        const predictionColumns = [];
        for (let colIndex = 0; colIndex < gridSize; colIndex++) {
            const colInputs = [];
            const x = colIndex / gridSize;
            for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
                const y = (gridSize - rowIndex) / gridSize;
                colInputs.push([x, y]);
            }

            const colPredictions = model.predict(tf.tensor2d(colInputs));
            predictionColumns.push(colPredictions);
        }
        const valuesTensor = tf.stack(predictionColumns);

        const normalizedTicksTensor = tf.linspace(0, 1, gridSize);
        const xTicksTensor = denormalize(normalizedTicksTensor,
            normalizedFeature.min[0], normalizedFeature.max[0]);
        const yTicksTensor = denormalize(normalizedTicksTensor.reverse(),
            normalizedFeature.min[1], normalizedFeature.max[1]); 
        
        return [valuesTensor.array(), xTicksTensor.array(), yTicksTensor.array()];
    });

    const values = await valuePromise;
    const xTicks = await xTicksPromise;
    const yTicks = await yTicksPromise;
    const xTickLabels = xTicks.map(v => (v/1000).toFixed(1) + "k sqft");
    const yTickLabels = yTicks.map(v => "$" + (v/1000).toFixed(0) + "k");
    const data = {
        values,
        xTickLabels,
        yTickLabels
    };
    

    tfvis.render.heatmap({
            name: `${name} local`,
            tab: "Predictions"
        },
        data, {
            height: size
        }
    );
    tfvis.render.heatmap({
            name: `${name} (full domain)`,
            tab: "Predictions"
        },
        data, {
            height: size,
            domain: [0, 1]
        }
    );
}

function normalize(tensor, previousMin = null, previousMax = null) {
    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];

    if (featureDimensions && featureDimensions > 1) {
        const features = tf.split(tensor, featureDimensions, 1);

        const normalizedFeatures = features.map((featureTensor, i) => 
            normalize(featureTensor,
                previousMin? previousMin[i] : null,
                previousMax? previousMax[i] : null
            )
        );

        const returnTensor = tf.concat(normalizedFeatures.map(f => f.tensor), 1);
        const min = normalizedFeatures.map(f => f.min);
        const max = normalizedFeatures.map(f => f.max);
        return { tensor: returnTensor, min, max };
    } else {
        const min = previousMin || tensor.min();
        const max = previousMax || tensor.max();
        console.log("min: ", min.print(), ", max: ", max.print());
        const normalizedTensor = tensor.sub(min).div(max.sub(min));
        return {
            tensor: normalizedTensor,
            min,
            max
        };
    }

    
}

function denormalize(tensor, min, max) {
     const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];

    if (featureDimensions && featureDimensions > 1) {
        const features = tf.split(tensor, featureDimensions, 1);

        const denormalizedFeatures = features.map((featureTensor, i) => 
            denormalize(featureTensor, min[i], max[i])            
        );

        const returnTensor = tf.concat(denormalizedFeatures.map(f => f.tensor), 1);        
        return returnTensor;
    } else {
        const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
        return denormalizedTensor;    
    }    
}

let model;
function createModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        useBias: true,
        activation: 'sigmoid', // 'linear', // 
        inputDim: 2
    }));   
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'sigmoid', // 'linear', //         
    }));   
    compileModel(model);

    return model;
}

function compileModel(model) {
    const optimizer = tf.train.adam(); // tf.train.sgd(0.1); // .1
    model.compile({
        loss: 'binaryCrossentropy', // 'meanSquaredError',
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
            onEpochEnd, //, onBatchEnd
            onEpochBegin: async function() {                
                const layer = model.getLayer(undefined, 0);
                tfvis.show.layer({ name: "Layer 1", tab: "Visor" }, layer);
                //await plotPredictionLine();
                await plotPredictionHeadmap();
            }
        }
    });
}

let points;
let normalizedFeature, normalizedLabel;
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;
async function run() {
    await tf.ready();

    // Import from csv
    const houseSalesDataset = tf.data.csv("https://ijianhuang.github.io/HouseSales/kc_house_data.csv")

    const className = "waterfront"; // "bedrooms"; // 
    // Extract x and y from data set to plot
    const pointsDataset = houseSalesDataset.map(record => ({
        x: record.sqft_living,
        y: record.price,
        class: record.waterfront // record[className] > 2? "3+" : record[className] // 
    }));
    points = await pointsDataset.toArray();
    if (points.length % 2 !== 0) {
        points.pop();
    }
    tf.util.shuffle(points); // attn: in-place shuffle 
    plotClasses(points, className);
    //plot(points, "Square Feet");

    // features (inputs)
    const featureValues = await points.map(p => [p.x, p.y]);
    const featureTensor = tf.tensor2d(featureValues);

    // labels (outputs)
    const labelValues = await points.map(p => p.class);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // normalize features and labels
    normalizedFeature = normalize(featureTensor);
    normalizedLabel = normalize(labelTensor);

    console.log(normalizedFeature);
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