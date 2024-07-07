const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let painting = false;

function startPosition(e) {
    painting = true;
    draw(e);
}

function endPosition() {
    painting = false;
    ctx.beginPath();
}

function draw(e) {
    if (!painting) return;
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);

function submitDrawing() {
    const dataURL = canvas.toDataURL('image/png');
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerText = `Predicted digit: ${data.digit}`;
            displayProbabilities(data.probabilities);
        });
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function displayProbabilities(probabilities) {
    const container = document.getElementById('probabilities');
    container.innerHTML = '';
    probabilities.forEach((prob, index) => {
        const probContainer = document.createElement('div');
        probContainer.className = 'probability-container';

        const label = document.createElement('div');
        label.className = 'probability-label';
        label.innerText = index;

        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar-container';

        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        bar.style.width = `${prob * 100}%`;

        const percentage = document.createElement('span');
        percentage.className = 'probability-percentage';
        percentage.innerText = ` ${(prob * 100).toFixed(2)}%`;

        barContainer.appendChild(bar);
        probContainer.appendChild(label);
        probContainer.appendChild(barContainer);
        probContainer.appendChild(percentage);

        container.appendChild(probContainer);
    });
}

// Clear the canvas on load
clearCanvas();
