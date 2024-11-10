class FeaturesAndLabelsBuilder {
    #slicePositionStart = 0;
    #slicePositionEnd = 0;

    constructor() {
        //this.config(datasetDescription, options);
    }
    
    config(datasetDescription, options) {
        this.datasetDescription = datasetDescription;
        this.datasetUrl = this.datasetDescription.datasetUrl;
        this.featureFields = this.datasetDescription.featureFields;
        this.labelFields = this.datasetDescription.labelFields;
        this.datasetSize = 0;
        this.#slicePositionStart = 0;
        this.#slicePositionEnd = 0;        
        this.options = Object.assign({
            shuffle: true,
            batchSize: 100
        }, options);
    }

   

    async loadSelectedFields() {
        // this.featureFields = featureFields;
        // this.labelFields = labelFields;
        this.datasetCSV = tf.data.csv(this.datasetUrl);          
        this.mapppedDataset = this.datasetCSV.map(record => { 
            const row = [];
            this.featureFields.forEach(p => row.push(record[p]));
            this.labelFields.forEach(p => row.push(record[p]));            
            return row;
        });
        this.mapppedDatasetArray = await this.mapppedDataset.toArray();  
        if (this.options.shuffle === true) {
            tf.util.shuffle(this.mapppedDatasetArray); // attn: in-place shuffle
        }         
        this.datasetSize = this.mapppedDatasetArray.length;
        this.numberOfBatches = Math.floor(this.datasetSize / this.options.batchSize);
        return this;
    }        

    getBatch(batchNumber) {
        this.#slicePositionStart = (batchNumber  - 1) * this.options.batchSize;
        this.#slicePositionEnd = this.#slicePositionStart + this.options.batchSize;
        this.selectedDatasetArray = this.mapppedDatasetArray.slice(this.#slicePositionStart, this.#slicePositionEnd);
        
        this.featuresAndLabels = tf.tensor2d(this.selectedDatasetArray);
        [this.features, this.labels] = 
            tf.split(this.featuresAndLabels, [this.featureFields.length, this.labelFields.length], 1);
        
        return [this.features, this.labels];
    }
}