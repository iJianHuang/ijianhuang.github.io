class KNN {
    constructor(featuresAndLabelsBuilder, datasetDescription, options = {}) {    
        this.debugInfo = "";
        this.topKMseValues1dHistory = tf.tensor1d([]);
        this.featuresHistory = [];
        this.labelsHistory = [];
        ///this.topKLabels2dHistory = tf.tensor2d([-1,-1], [-1,-1]);  
        this.featuresAndLabelsBuilder = featuresAndLabelsBuilder;
        this.datasetDescription = datasetDescription;
        this.options = Object.assign({
            shuffle: true,
            batchSize: 100
        }, options);        
    }
    

    #addFeaturesAndLabels(features, labels) 
    {
        this.features = features;
        this.labels = labels;
        if (this.features === null || this.labels === null) {
            return;
        } 
        
        this.featuresHistory.push(this.features);
        this.labelsHistory.push(this.labels);
        this.#standardize();
    }

    #standardize() {
       
        const {mean, variance} = tf.moments(this.features, 0); //  The dimension(s) along with to compute
        this.mean = mean;
        this.variance = variance;
        this.standardizedFeatures = this.features
            .sub(this.mean)
            .div(this.variance.pow(.5));
    }

    async predict(sampleValues, topK) {
       

        this.featuresAndLabelsBuilder.config(this.datasetDescription, this.options);
        await this.featuresAndLabelsBuilder.loadSelectedFields();

        let price = 0;
        for (let i = 1; i <= this.featuresAndLabelsBuilder.numberOfBatches; i++) {
            let [features, labels] = this.featuresAndLabelsBuilder.getBatch(i);
            this.#addFeaturesAndLabels(features, labels);
            price = this.#learn(sampleValues, topK);
            this.debugInfo += `Price ${i}:  ${price} ` + '\r\n';
        }          
        return  price;
    }

    #learn(sampleValues, topK) {
        if (this.features !== null && this.labels !== null) {
            this.#learnThisBatch(sampleValues, topK);
        }         

        const {values, indices} = tf.topk(this.topKMseValues1dHistory, topK);    
        this.topKMseValues1d = values;
        this.topKMseIndices1d = indices;  

        this.topKLabels2d = this.topKLabels2dHistory.gather(indices);
        // reshape(-1) or flatten
        return  this.topKLabels2d.sum().dataSync()[0] / topK;
    }

    #learnThisBatch(sampleValues, topK) {
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
