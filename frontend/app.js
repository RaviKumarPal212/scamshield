const form = document.querySelector('#analyze-form');
const messageInput = document.querySelector('#message');
const count = document.querySelector('#character-count');
const button = document.querySelector('#analyze-button');
const clearButton = document.querySelector('#clear-button');
const errorBox = document.querySelector('#form-error');
const result = document.querySelector('#result');
const serviceState = document.querySelector('#service-state');

function setServiceState(ready) {
  serviceState.textContent = ready ? 'Service ready' : 'Service unavailable';
  serviceState.classList.toggle('ready', ready);
  serviceState.classList.toggle('offline', !ready);
}

async function checkService() {
  try {
    const response = await fetch('/health', { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error('Service is not ready');
    setServiceState(true);
  } catch {
    setServiceState(false);
  }
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function renderResult(data) {
  const flagged = data.classification === 'spam_like';
  document.querySelector('#result-tag').textContent = flagged ? 'Potentially suspicious' : 'No strong spam signal';
  document.querySelector('#result-tag').className = `result-tag ${flagged ? 'flagged' : 'not-flagged'}`;
  document.querySelector('#result-title').textContent = flagged ? 'Take a moment before acting.' : 'No strong signal was found.';
  document.querySelector('#result-summary').textContent = flagged
    ? 'The model marked this message as spam-like. That can include unwanted messages; it does not establish that the sender is a fraudster.'
    : 'This message did not cross the model’s screening threshold. Harmful messages can still be missed.';

  const probability = Math.max(0, Math.min(1, Number(data.spam_probability)));
  document.querySelector('#probability-value').textContent = `${(probability * 100).toFixed(1)}%`;
  const meter = document.querySelector('#probability-meter');
  meter.style.width = `${probability * 100}%`;
  meter.classList.toggle('high', probability >= 0.5);
  document.querySelector('#threshold-note').textContent = `Decision threshold: ${(Number(data.threshold) * 100).toFixed(1)}% · this model output is not a verified probability of fraud.`;

  const list = document.querySelector('#signal-list');
  list.replaceChildren();
  document.querySelector('#signal-count').textContent = `${data.signal_count} found`;
  if (!data.signals.length) {
    const empty = document.createElement('li');
    empty.className = 'no-signals';
    empty.textContent = 'No listed rule indicators were detected. This does not make the message safe.';
    list.append(empty);
  } else {
    for (const signal of data.signals) {
      const item = document.createElement('li');
      item.className = 'signal-item';
      const title = document.createElement('strong');
      title.textContent = signal.title;
      const explanation = document.createElement('p');
      explanation.textContent = signal.explanation;
      item.append(title, explanation);
      list.append(item);
    }
  }
  document.querySelector('#result-disclaimer').textContent = data.disclaimer;
  result.hidden = false;
  result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

messageInput.addEventListener('input', () => {
  count.textContent = `${messageInput.value.length} / 5000`;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  result.hidden = true;
  const message = messageInput.value;
  if (!message.trim()) {
    showError('Enter a message to screen.');
    messageInput.focus();
    return;
  }

  button.disabled = true;
  button.querySelector('span:first-child').textContent = 'Checking…';
  try {
    const response = await fetch('/v1/analyze/sms', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? 'Please check the message and try again.' : data.detail;
      throw new Error(typeof detail === 'string' ? detail : 'The message could not be checked.');
    }
    renderResult(data);
  } catch (error) {
    showError(error instanceof TypeError ? 'Could not reach the local service. Make sure it is running, then try again.' : error.message);
    setServiceState(false);
  } finally {
    button.disabled = false;
    button.querySelector('span:first-child').textContent = 'Check message';
  }
});

clearButton.addEventListener('click', () => {
  form.reset();
  count.textContent = '0 / 5000';
  errorBox.hidden = true;
  result.hidden = true;
  messageInput.focus();
});

checkService();
