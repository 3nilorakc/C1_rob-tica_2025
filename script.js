
let model, labels;

const preview = document.getElementById('preview');
const result = document.getElementById('result');
const loading = document.getElementById('loading');

// Carrega modelo e labels
async function loadModel() {
  try {
    loading.textContent = '🔄 Carregando modelo...';
    model = await tf.loadLayersModel('model.json');
    const metadata = await fetch('metadata.json').then(res => res.json());
    labels = metadata.labels;
    loading.textContent = '';
    console.log('Modelo carregado com sucesso.');
  } catch (err) {
    loading.textContent = '❌ Erro ao carregar o modelo.';
    console.error(err);
  }
}

document.getElementById('imageInput').addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (event) {
    preview.src = event.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
});

async function predict() {
  if (!model || !labels) {
    alert('O modelo ainda está carregando!');
    return;
  }

  if (!preview.src || preview.style.display === 'none') {
    alert('Por favor, selecione uma imagem primeiro!');
    return;
  }

  loading.textContent = '🔎 Analisando imagem...';
  result.innerHTML = '';

  const imgTensor = tf.browser.fromPixels(preview)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .expandDims();

  const predictions = await model.predict(imgTensor).array();
  const probs = predictions[0];

  const top3 = probs
    .map((p, i) => ({ label: labels[i], prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3);

  result.innerHTML = '<strong>🔍 Top 3 previsões:</strong><br><br>' +
    top3.map(p =>
      `${p.label} – ${(p.prob * 100).toFixed(2)}%`
    ).join('<br>');

  loading.textContent = '';
}

loadModel();
