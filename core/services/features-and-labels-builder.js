class FeaturesAndLabelsBuilder {
    #unprocessedRecordsSize = 0;    

    constructor(datasetUrl, options = {}) 
    {
        this.datasetUrl = datasetUrl;
        this.datasetSize = 0;
        this.slicePositionStart = 0;  
        this.slicePositionEnd = 0;  
        this.#unprocessedRecordsSize = 0  
        this.options = Object.assign({
            shuffle: true
        }, options);
    }
    
    async loadSelectedFields(featureFields, labelFields) {
        this.featureFields = featureFields;
        this.labelFields = labelFields;
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
        return this;
    }

    getDatasetSize() {
        return this.datasetSize;
    }

    getUnprocessedRecordsSize() {
        this.#unprocessedRecordsSize = this.datasetSize - this.slicePositionStart;
        return this.#unprocessedRecordsSize;
    }

    getNextBatch(batchSize) {
        let actualBatchSize = 0;
        this.#unprocessedRecordsSize = this.datasetSize - this.slicePositionStart;
        if (this.#unprocessedRecordsSize === 0) {
            return [tf.tensor2d([[]]), tf.tensor2d([[]])];
        }

        if (batchSize <= this.#unprocessedRecordsSize) {
            actualBatchSize = batchSize;
        } else {
            actualBatchSize = this.#unprocessedRecordsSize;
        }
        this.slicePositionEnd = this.slicePositionStart + actualBatchSize;
        this.selectedDatasetArray = this.mapppedDatasetArray.slice(this.slicePositionStart, this.slicePositionEnd);
        this.slicePositionStart += actualBatchSize;
        this.#unprocessedRecordsSize = this.datasetSize - this.slicePositionStart;
    
        this.featuresAndLabels = tf.tensor2d(this.selectedDatasetArray);
        [this.features, this.labels] = 
            tf.split(this.featuresAndLabels, [this.featureFields.length, this.labelFields.length], 1);
        
        return [this.features, this.labels];
    }
}