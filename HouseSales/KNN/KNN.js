class KNN 
{
    constructor() 
    {    
        this.topKMseValues1dHistory = tf.tensor1d([]);
        ///this.topKLabels2dHistory = tf.tensor2d([-1,-1], [-1,-1]);          
    }

    reset() {
        this.topKMseValues1dHistory = tf.tensor1d([]);
        this.topKLabels2dHistory = null;
    }

    addFeaturesAndLabels(features, labels) 
    {
        this.features = features;
        this.labels = labels;
        this.#buildFeaturesAndLabel();
    }

    #buildFeaturesAndLabel() {
       
        const {mean, variance} = tf.moments(this.features, 0); //  The dimension(s) along with to compute
        this.mean = mean;
        this.variance = variance;
        this.standardizedFeatures = features
            .sub(this.mean)
            .div(this.variance.pow(.5));
    }

    predict(sampleValues, topK) {
        this.#train(sampleValues, topK);

        const {values, indices} = tf.topk(this.topKMseValues1dHistory, topK);    
        this.topKMseValues1d = values;
        this.topKMseIndices1d = indices;  

        this.topKLabels2d = this.topKLabels2dHistory.gather(indices);
        // reshape(-1) or flatten
        return  this.topKLabels2d.sum().dataSync()[0] / topK;
    }

    #train(sampleValues, topK) {
        const predictionPointFeature1d = tf.tensor1d(sampleValues);
        const standardizedPredictionPointFeature1d = predictionPointFeature1d
            .sub(this.mean)
            .div(this.variance.pow(.5));

        const mse1d = this.standardizedFeatures
            .sub(standardizedPredictionPointFeature1d)
            .pow(2)
            .sum(1)
            .pow(.5);
            //.expandDims(1);

        const {values, indices} = tf.topk(mse1d.mul(-1), topK);    
        this.topKMseValues1dThisTime = values;
        this.topKMseIndices1dThisTime = indices;   
        this.topKLabels2dThisTime = this.labels.gather(indices);

        this.topKMseValues1dHistory = this.topKMseValues1dHistory.concat(values);
        if (this.topKLabels2dHistory === undefined || this.topKLabels2dHistory === null) {
            this.topKLabels2dHistory = this.topKLabels2dThisTime;
        } else {
            this.topKLabels2dHistory = this.topKLabels2dHistory.concat(this.topKLabels2dThisTime, 0);
        }

    }
}


class FeaturesAndLabelsBuilder {
    #unprocessedRecordsSize = 0;

    constructor(datasetUrl, options = {}) 
    {
        this.DatasetUrl = datasetUrl;
        this.DatasetSize = 0;
        this.slicePositionStart = 0;  
        this.slicePositionEnd = 0;  
        this.#unprocessedRecordsSize = 0  
        this.options = Object.assign({
            shuffle: true
        }, options);
    }
    
    async loadSelectedFields(featureFields, labelFields) {
        this.DatasetCSV = tf.data.csv(this.DatasetUrl);          
        this.MapppedDataset = this.DatasetCSV.map(record => { 
            const row = [];
            featureFields.forEach(p => row.push(record[p]));
            labelFields.forEach(p => row.push(record[p]));            
            return row;
        });
        this.MapppedDatasetArray = await this.MapppedDataset.toArray();  
        if (this.options.shuffle === true) {
            tf.util.shuffle(this.MapppedDatasetArray); // attn: in-place shuffle
        }         
        this.DatasetSize = this.MapppedDatasetArray.length;
        return this;
    }

    getDatasetSize() {
        return this.DatasetSize;
    }

    getUnprocessedRecordsSize() {
        this.#unprocessedRecordsSize = this.DatasetSize - this.slicePositionStart;
        return this.#unprocessedRecordsSize;
    }

    getNextBatch(batchSize) {
        let actualBatchSize = 0;
        this.#unprocessedRecordsSize = this.DatasetSize - this.slicePositionStart;
        if (this.#unprocessedRecordsSize === 0) {
            return [tf.tensor2d([[]]), tf.tensor2d([[]])];
        }

        if (batchSize <= this.#unprocessedRecordsSize) {
            actualBatchSize = batchSize;
        } else {
            actualBatchSize = this.#unprocessedRecordsSize;
        }
        this.slicePositionEnd = this.slicePositionStart + actualBatchSize;
        this.SelectedDatasetArray = this.MapppedDatasetArray.slice(this.slicePositionStart, this.slicePositionEnd);
        this.slicePositionStart += actualBatchSize;
        this.#unprocessedRecordsSize = this.DatasetSize - this.slicePositionStart;
    
        this.FeaturesAndLabels = tf.tensor2d(this.SelectedDatasetArray);
        [this.Features, this.Labels] = 
            tf.split(this.FeaturesAndLabels, [featureFields.length, labelFields.length], 1);
        
        return [this.Features, this.Labels];
    }

}

const featureFields = ["lat", "long"];
const labelFields = ["price"];
const builder = new FeaturesAndLabelsBuilder("https://ijianhuang.github.io/HouseSales/kc_house_data.csv");
await builder.loadSelectedFields(featureFields, labelFields);

const [features, labels] = builder.getNextBatch(5000);

const knn = new KNN();
knn.addFeaturesAndLabels(features, labels);
const price = knn.predict([47.38, -122.44], 3);

const [features2, labels2] = builder.getNextBatch(5000);
knn.addFeaturesAndLabels(features2, labels2);
const price2 = knn.predict([47.38, -122.44], 3);

const [features3, labels3] = builder.getNextBatch(5000);
knn.addFeaturesAndLabels(features3, labels3);
const price3 = knn.predict([47.38, -122.44], 3);

const [features4, labels4] = builder.getNextBatch(5000);
knn.addFeaturesAndLabels(features4, labels4);
const price4 = knn.predict([47.38, -122.44], 3);
// knn.topKLabels2dHistory.concat(knn.topKMseValues1dHistory.expandDims(1), 1).print();
