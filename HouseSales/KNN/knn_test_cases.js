
async function runTests() {
    let debugInfo = "Loading test cases ... ";
    document.getElementById("testing-status").innerHTML = debugInfo;
    
    const datasetDescription = {
        datasetUrl: "https://ijianhuang.github.io/assets/kc_house_data_1000.csv",
        featureFields: ["lat", "long"],
        labelFields: ["price"] 
    };
    const options =  { batchSize: 50 };
    const builder = new FeaturesAndLabelsBuilder();   

    const knn = new KNN(builder, datasetDescription, options);
    let price = await knn.predict([47.38, -122.44], 3);
    

    
    const datasetSize = knn.featuresAndLabelsBuilder.datasetSize;
    debugInfo = "dataset size is 1000: " + (datasetSize === 1000) + '\r\n';
    document.getElementById("testing-status").innerHTML = debugInfo;

    
    debugInfo += `numberOfBatches is 10: ` + (knn.featuresAndLabelsBuilder.numberOfBatches === 10) + '\r\n';   
    debugInfo += `Price: ${price}`  + '\r\n \r\n \r\n';
    debugInfo += knn.debugInfo;   

    document.getElementById("testing-status").innerHTML = debugInfo + `Test done`;

}

