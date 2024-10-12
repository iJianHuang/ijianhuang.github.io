class KNN 
{
    constructor(datasetUrl) 
    {
        this.datasetUrl = datasetUrl;
        this.datasetCSV = tf.data.csv(this.datasetUrl);

    }

    async buildFeaturesAndLabel(featureFields, labelField) {
        this.mapppedDataset = this.datasetCSV.map(record => { 
            const row = [];
            featureFields.forEach(p => row.push(record[p]));
            row.push(record[labelField]);
            return row;
        });
        this.mapppedDatasetArray = await this.mapppedDataset.toArray();    
        tf.util.shuffle(this.mapppedDatasetArray); // attn: in-place shuffle
        this.selectedDatasetArray = this.mapppedDatasetArray.slice(0, 5000);
    
        this.featuresAndLabel = tf.tensor2d(this.selectedDatasetArray);
        [this.features, this.label] = tf.split(this.featuresAndLabel, [featureFields.length, 1], 1);
        //pointsTensor.slice([1,0], [3,4]).print();
    
        const {mean, variance} = tf.moments(features, 0); //  The dimension(s) along with to compute
        this.mean = mean;
        this.variance = variance;
        this.standardizedFeatures = features
            .sub(this.mean)
            .div(this.variance.pow(.5));

        return this;
    }

    predict(values) {
        //const predictionPoint = [47.38, -122.44, 1970]; //, 705000];
        const predictionPointFeature1d = tf.tensor1d(values);
        const standardizedPredictionPointFeature1d = predictionPointFeature1d.sub(mean).div(variance.pow(.5));

        const mse1d = this.standardizedFeatures
            .sub(standardizedPredictionPointFeature1d)
            .pow(2)
            .sum(1)
            .pow(.5);
            //.expandDims(1);

        const {values, indices} = tf.topk(mse1d.mul(-1), 3);    
        values.print();
        indices.print();

        const topKIndexes = indices.arraySync();
        const topKIndexesTensor1d = tf.tensor1d(topKIndexes, 'int32');
        const topKValues = pointsLabels.gather(topKIndexesTensor1d);

        return topKValues.sum().dataSync()[0] / topKValues.shape[0];
    }



}