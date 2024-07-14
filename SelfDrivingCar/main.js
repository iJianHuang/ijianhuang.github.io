const carCanvas=document.getElementById("carCanvas");
carCanvas.width=200;
const networkCanvas=document.getElementById("networkCanvas");
networkCanvas.width=300;

const carCtx = carCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");

const road=new Road(carCanvas.width/2,carCanvas.width*0.9);

const N=100;
const cars=generateCars(N);
let bestCar=cars[0];

localStorage.setItem("bestBrain",
        '{"levels":[{"inputs":[0,0,0,0.3446579259786464,0.645331425277923],"outputs":[0,0,1,0,0,1],"biases":[0.466492516864187,0.4100775862911721,-0.4547801937248566,-0.18708767815040644,0.5075025457832779,-0.6570961288536572],"weights":[[-0.40323916628420275,-0.7249348332118016,-0.10574189384269989,-0.4780763525280275,-0.14040942667964432,-0.21109615277371097],[-0.184845449287651,-0.190946004532615,-0.16199482866050122,0.443962248693768,-0.34694404997770767,-0.6055301954551966],[0.2813146359482699,0.4283822732356033,-0.06358038060883414,-0.263553183261098,0.31760468345907544,-0.08153436647776235],[0.10392595678294053,0.7101625525263306,0.14132915442120703,-0.16358022995478202,-0.19433509118547182,-0.19987380389736692],[0.00698125789407264,0.20568659197463407,0.1096751132126953,-0.49347877501847615,-0.46673090703651554,-0.012886216510876902]]},{"inputs":[0,0,1,0,0,1],"outputs":[1,1,1,0],"biases":[-0.4570251097940761,-0.5258951326561848,-0.5755951143312096,0.40616264359057985],"weights":[[-0.1940547294266221,0.4064878920765438,0.06343420768656094,0.1982023915940004],[0.68998913144668,-0.12826954060715812,-0.6316086589099199,-0.362697784417402],[0.6160229022837823,0.3076870580962486,-0.3571222100725432,-0.2082087991135273],[0.45955737905888827,-0.5484442692019883,0.19722494762698636,-0.6241979142305059],[0.13059779669307275,0.6497252873356053,-0.3151111552042579,0.15388130376012654],[-0.2095796433148313,-0.5349018515304337,0.2624868016482013,-0.47300168368156414]]}]}');
if(localStorage.getItem("bestBrain")){
    for(let i=0;i<cars.length;i++){
        cars[i].brain=JSON.parse(
            localStorage.getItem("bestBrain"));
        if(i!=0){
            NeuralNetwork.mutate(cars[i].brain,0.3);
        }
    }
}

const traffic=[
    new Car(road.getLaneCenter(1),-100,30,50,"DUMMY",2,getRandomColor()),
    new Car(road.getLaneCenter(0),-300,30,50,"DUMMY",2,getRandomColor()),
    new Car(road.getLaneCenter(2),-300,30,50,"DUMMY",2,getRandomColor()),
    new Car(road.getLaneCenter(0),-500,30,50,"DUMMY",2,getRandomColor()),
    new Car(road.getLaneCenter(1),-500,30,50,"DUMMY",2,getRandomColor()),
    new Car(road.getLaneCenter(1),-700,30,50,"DUMMY",2,getRandomColor()),
    new Car(road.getLaneCenter(2),-700,30,50,"DUMMY",2,getRandomColor()),
];

animate();

function save(){
    localStorage.setItem("bestBrain",
        JSON.stringify(bestCar.brain));
}

function discard(){
    localStorage.removeItem("bestBrain");
}

function generateCars(N){
    const cars=[];
    for(let i=1;i<=N;i++){
        cars.push(new Car(road.getLaneCenter(1),100,30,50,"AI"));
    }
    return cars;
}

function animate(time){
    for(let i=0;i<traffic.length;i++){
        traffic[i].update(road.borders,[]);
    }
    for(let i=0;i<cars.length;i++){
        cars[i].update(road.borders,traffic);
    }
    bestCar=cars.find(
        c=>c.y==Math.min(
            ...cars.map(c=>c.y)
        ));

    carCanvas.height=window.innerHeight;
    networkCanvas.height=window.innerHeight;

    carCtx.save();
    carCtx.translate(0,-bestCar.y+carCanvas.height*0.7);

    road.draw(carCtx);
    for(let i=0;i<traffic.length;i++){
        traffic[i].draw(carCtx);
    }
    carCtx.globalAlpha=0.2;
    for(let i=0;i<cars.length;i++){
        cars[i].draw(carCtx);
    }
    carCtx.globalAlpha=1;
    bestCar.draw(carCtx,true);

    carCtx.restore();

    networkCtx.lineDashOffset=-time/50;
    Visualizer.drawNetwork(networkCtx,bestCar.brain);
    requestAnimationFrame(animate);
}