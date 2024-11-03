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
        this.standardizedFeatures = this.features
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
