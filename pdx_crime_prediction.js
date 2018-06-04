
const modelNeighborhood = tf.sequential();
const modelOffenseType = tf.sequential();
const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);
const refresh_period_ms = 10000;
const neighborhoodMap = new Map();
const offenseTypeMap = new Map();
const timeDivisor = 1000000000000;
const neighborhoodDivisor = 1000000000;
const offenseTypeDivisor  = 1000000000;
const datasetReducer =  2;
const useMap = true;

let xdate;
let yneighborhood;
let yoffensetype;


document.getElementById('status').textContent = 'Loading data...';
fetch("pdx_crime_2018.json").then((response) => {
	return response.json();
}).then((dataset) => {
	document.getElementById('status').textContent = 'Data loaded...';
	build_model(
		dataset.map(d => {
			d["datetime"] = new Date(d["Occur Date"] + " " + pad(d["Occur Time"]).replace(/\b(\d{1,2})(\d{2})/g, '$1:$2'));
			d["datetime_ms"] = d["datetime"].getTime();
			return d;
		}).sort( (a,b) => {
			return a["datetime_ms"] - b["datetime_ms"];
		})
	);
});

function build_model(dataset) {

	document.getElementById('status').textContent = 'Building model...';

	modelNeighborhood.add(tf.layers.dense({units: 1, inputShape: [1]}));;
	modelNeighborhood.compile({loss: 'meanSquaredError', optimizer: optimizer});
	modelOffenseType.add(tf.layers.dense({units: 1, inputShape: [1]}));
	modelOffenseType.compile({loss: 'meanSquaredError', optimizer: optimizer});

	dataset = dataset.slice(0,dataset.length/datasetReducer);
	dataset.forEach(d => neighborhoodMap.set(d['Neighborhood'].hashCode(), d['Neighborhood']) );
	dataset.forEach(d => offenseTypeMap.set(d['Offense Type'].hashCode(), d['Offense Type']) );

	const xset = dataset.map( d => d['datetime_ms'] / timeDivisor );
	const ysetNeighborhood = dataset.map( d => d['Neighborhood'].hashCode() / neighborhoodDivisor );
	const ysetOffenseType = dataset.map( d => d['Offense Type'].hashCode() / offenseTypeDivisor );

	// console.log(xset.findIndex(isNaN));
	// console.log(Math.max(...xset));
	// console.log(Math.max(...ysetNeighborhood));
	// console.log(Math.min(...xset));
	// console.log(Math.min(...ysetNeighborhood));

	xdate = tf.tensor2d(xset, [dataset.length, 1]);
	yneighborhood = tf.tensor2d(ysetNeighborhood, [dataset.length, 1]);
	yoffensetype = tf.tensor2d(ysetOffenseType, [dataset.length, 1]);
  
	fit();

	setInterval(() => {
		fit();
	}, refresh_period_ms);
}

function fit() {
	document.getElementById('status').textContent = 'Predicting...';

	const now_ms = new Date().getTime();
	const future_time_variance = Math.floor(Math.random() * (3600000 - 600000 + 1) ) + 600000;
	const future_time_ms = new Date(now_ms + future_time_variance).getTime();

	Promise.all([
		modelNeighborhood.fit(xdate, yneighborhood, { shuffle: true, epochs: 1 }),
		modelOffenseType.fit(xdate, yoffensetype, { shuffle: true, epochs: 1 }),
	]).then(() => {
			predict(future_time_ms/timeDivisor);
	});
}

function predict(datetime_ms) {

	const n = modelNeighborhood.predict(tf.tensor2d([datetime_ms], [1, 1])).buffer().values[0]
	const neighborhoodKeys = [...neighborhoodMap.keys()], neighborGoal = n * neighborhoodDivisor;
	const closestNeighborhood = neighborhoodKeys.reduce( (prev, curr) => {
	  return (Math.abs(curr - neighborGoal) < Math.abs(prev - neighborGoal) ? curr : prev);
	});

	const o = modelOffenseType.predict(tf.tensor2d([datetime_ms], [1, 1])).buffer().values[0]
	const offensetypeKeys = [...offenseTypeMap.keys()], offenseGoal = o * offenseTypeDivisor;
	const closestOffenseType = offensetypeKeys.reduce( (prev, curr) => {
	  return (Math.abs(curr - offenseGoal) < Math.abs(prev - offenseGoal) ? curr : prev);
	});

	console.log(neighborhoodMap.get(closestNeighborhood) + " " + offenseTypeMap.get(closestOffenseType));
	console.log(n + " " + o);

	if (useMap) {	
		let location = encodeURI(neighborhoodMap.get(closestNeighborhood));
		const iframeUrl = `https://www.google.com/maps/embed/v1/place?q=${location}%2C%20Portland%2C%20OR&key=AIzaSyCOzLKykk4I2pO1GcNzKw6OaS3M_7wEp5o`
		document.getElementById('map').src = iframeUrl;
		if (neighborhoodMap.get(closestNeighborhood) === null || neighborhoodMap.get(closestNeighborhood) === "" )
			location = "Portland, OR";
		else 
			location = neighborhoodMap.get(closestNeighborhood);

		document.getElementById('status').textContent = 
			"Prediction: " + offenseTypeMap.get(closestOffenseType) + 
			" in " + location + 
			" at " + new Date(datetime_ms * timeDivisor).toLocaleString();
	}

}

String.prototype.hashCode = function() {
  var hash = 0, i, chr;
  if (this.length === 0) return hash;
  for (i = 0; i < this.length; i++) {
	chr   = this.charCodeAt(i);
	hash  = ((hash << 5) - hash) + chr;
	hash |= 0; // Convert to 32bit integer
  }
  return hash;
};

function pad(val) {
	const str = "" + val;
	const pad = "0000"
	return pad.substring(0, pad.length - str.length) + str;
}
