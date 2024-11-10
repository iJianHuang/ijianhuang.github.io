class BatchManager {    
    constructor(numberOfRecords) { 
        this.numberOfRecords = numberOfRecords;  
        this.slicePositionStart = 0;  
        this.slicePositionEnd = 0;  
        this.actualBatchSize = 0;   
        this.numberOfRecordsProcessed = 0;
        this.numberOfRecordsNotProcessedYet = 0;  
    }
}