

async function runTests() {
    let debugInfo = "Loading test cases ... ";
    document.getElementById("testing-status").innerHTML = debugInfo;
    const featureFields = ["lat", "long"];
    const labelFields = ["price"];
    let builder = null;    
    if (window.location.href.indexOf("file:///", 0) === 0) {
        builder = new FeaturesAndLabelsBuilder("../KNN/kc_house_data.csv");
    } else {
        builder = new FeaturesAndLabelsBuilder("https://ijianhuang.github.io/HouseSales/kc_house_data.csv");   
    }

    await builder.loadSelectedFields(featureFields, labelFields);
      

    const [features, labels] = builder.getNextBatch(5000);

    const knn = new KNN();
    knn.addFeaturesAndLabels(features, labels);
    const price = knn.predict([47.38, -122.44], 3);
    debugInfo = `Price 1:  ${price} ` + '\r\n';
    

    const [features2, labels2] = builder.getNextBatch(5000);
    knn.addFeaturesAndLabels(features2, labels2);
    const price2 = knn.predict([47.38, -122.44], 3);
    debugInfo += `Price 2:  ${price2} ` + '\r\n';

    const [features3, labels3] = builder.getNextBatch(5000);
    knn.addFeaturesAndLabels(features3, labels3);
    const price3 = knn.predict([47.38, -122.44], 3);
    debugInfo += `Price 3: ${price3} ` + '\r\n';

    document.getElementById("testing-status").innerHTML = debugInfo + `Test done`;

}

