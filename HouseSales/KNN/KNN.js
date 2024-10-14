class KNN 
{
    constructor(features, label) 
    {
        this.features = features;
        this.label = label;
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
        //const predictionPoint = [47.38, -122.44, 1970]; //, 705000];
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
        this.topKValues1d = values;
        this.topKIndices1d = indices;        

        const topKIndicesArray = this.topKIndices1d.arraySync();
        const topKIndexes1d = tf.tensor1d(topKIndicesArray, 'int32');
        const topKValues2d = this.label.gather(topKIndexes1d);

        return  topKValues2d.sum().dataSync()[0] / topK;
    }
}


async function buildFeaturesAndLabel() {
    const datasetCSV = tf.data.csv("https://ijianhuang.github.io/HouseSales/kc_house_data.csv");
    const featureFields = ["lat", "long"];
    const labelField = "price";
    const mapppedDataset = datasetCSV.map(record => { 
        const row = [];
        featureFields.forEach(p => row.push(record[p]));
        row.push(record[labelField]);
        return row;
    });
    const mapppedDatasetArray = await mapppedDataset.toArray();    
    tf.util.shuffle(mapppedDatasetArray); // attn: in-place shuffle
    const selectedDatasetArray = mapppedDatasetArray.slice(0, 5000);

    const featuresAndLabel = tf.tensor2d(selectedDatasetArray);
    const [features, label] = tf.split(featuresAndLabel, [featureFields.length, 1], 1);
   
    return [features, label];    
}

class FeaturesAndLabelsBuilder {
    constructor(datasetUrl) 
    {
        this.datasetUrl = datasetUrl;
        this.datasetCSV = tf.data.csv(this.datasetUrl);

    }

    async buildFeaturesAndLabels(featureFields, labelFields) {
        this.mapppedDataset = this.datasetCSV.map(record => { 
            const row = [];
            featureFields.forEach(p => row.push(record[p]));
            labelFields.forEach(p => row.push(record[p]));            
            return row;
        });
        this.mapppedDatasetArray = await this.mapppedDataset.toArray();    
        tf.util.shuffle(this.mapppedDatasetArray); // attn: in-place shuffle
        this.selectedDatasetArray = this.mapppedDatasetArray.slice(0, 5000);
    
        this.featuresAndLabel = tf.tensor2d(this.selectedDatasetArray);
        [this.features, this.labels] = tf.split(this.featuresAndLabel, [featureFields.length, labelFields.length], 1);
        
        return [this.features, this.labels];
    }

}

const featureFields = ["lat", "long"];
const labelFields = ["price"];
const builder = new FeaturesAndLabelsBuilder("https://ijianhuang.github.io/HouseSales/kc_house_data.csv");

const [features, labels] = await builder.buildFeaturesAndLabels(featureFields, labelFields);

const knn = new KNN(features, labels);
const price = knn.predict([47.38, -122.44], 3);

