(() => {
  const STORAGE_KEY = 'exoscan-prediction-shell-v1';
  const MAX_FEATURES = 12;

  function ready() {
    const page = document.body;
    if (!page.classList.contains('prediction-page')) {
      return;
    }

    const elements = {
      objectGrid: document.querySelector('[data-object-grid]'),
      cardTemplate: document.getElementById('data-card-template'),
      addButton: document.querySelector('[data-action="add-object"]'),
      importInput: document.querySelector('[data-action="import-object"] input[type="file"]'),
      selectionStatus: document.querySelector('[data-selection-status]'),
      chooseCardButton: document.querySelector('[data-action="choose-card"]'),
      singleUploadInput: document.querySelector('[data-action="upload-single"] input[type="file"]'),
      singleUploadList: document.querySelector('[data-upload-list="single"]'),
      bulkUploadInput: document.querySelector('[data-action="upload-bulk"] input[type="file"]'),
      bulkUploadList: document.querySelector('[data-upload-list="bulk"]'),
      bulkDownloadButton: document.querySelector('[data-action="download-bulk"]'),
      remoteLinkButton: document.querySelector('[data-action="link-remote"]'),
      runButton: document.querySelector('[data-action="run-prediction"]'),
      outputLog: document.querySelector('[data-output-log]'),
      objectLibrarySection: document.getElementById('object-library'),
      inspector: document.querySelector('[data-object-inspector]'),
      inspectorPanel: document.querySelector('[data-object-inspector] .object-inspector__panel'),
      inspectorBackdrop: document.querySelector('[data-object-inspector] .object-inspector__backdrop'),
      inspectorForm: document.querySelector('[data-inspector-form]'),
      inspectorModeLabel: document.querySelector('[data-inspector-mode]'),
      inspectorTitle: document.querySelector('[data-inspector-title]'),
      inspectorMeta: document.querySelector('[data-inspector-meta]'),
      inspectorFeatures: document.querySelector('[data-inspector-features]'),
      inspectorNameInput: document.querySelector('[data-inspector-name]'),
      inspectorSubtitleInput: document.querySelector('[data-inspector-subtitle]'),
      inspectorNotes: document.querySelector('[data-inspector-notes]'),
      inspectorHelper: document.querySelector('[data-inspector-helper]'),
      inspectorSaveButton: document.querySelector('[data-inspector-save]')
    };

    if (!elements.objectGrid || !elements.cardTemplate) {
      return;
    }

    const templateMetrics = extractTemplateMetrics(elements.cardTemplate);
    const storedState = loadStoredState();
    const initialCards = storedState && Array.isArray(storedState.cards) && storedState.cards.length
      ? sanitizeCards(storedState.cards, templateMetrics)
      : extractCardsFromDom(elements.objectGrid, templateMetrics);

    const state = {
      cards: initialCards,
      objectCounter: getHighestCounterFromCards(initialCards),
      selectedId: null,
      runnerSelectionId: null,
      uploads: {
        single: null,
        bulk: []
      },
      bulkResults: null, // Store bulk processing results for download
      logs: [],
      inspector: {
        activeCard: null,
        snapshot: null,
        mode: 'open',
        lastFocused: null,
        keydownHandler: null
      },
      templateMetrics,
      pendingIntroId: null,
      chooseMode: false,
      storageEnabled: typeof window.localStorage !== 'undefined'
    };

    if (storedState && storedState.selectedId) {
      state.selectedId = cardExists(initialCards, storedState.selectedId) ? storedState.selectedId : null;
    }
    if (!state.selectedId && initialCards.length) {
      state.selectedId = initialCards[0].id;
    }

    if (storedState && storedState.runnerSelectionId) {
      state.runnerSelectionId = cardExists(initialCards, storedState.runnerSelectionId) ? storedState.runnerSelectionId : null;
    }
    if (!state.runnerSelectionId) {
      state.runnerSelectionId = state.selectedId || null;
    }

    renderCards(elements, state);

    renderUploadList(elements.singleUploadList, [], { type: 'single' });
    renderUploadList(elements.bulkUploadList, [], { type: 'bulk' });

    initInspector(elements, state);
    initCardActions(elements, state);
    initAddObject(elements, state);
    initImport(elements, state);
    initSelection(elements, state);
    initChooseCard(elements, state);
    initSingleUpload(elements, state);
    initBulkUpload(elements, state);
    initBulkListRemoval(elements, state);
    initBulkDownload(elements, state);
    initRunAction(elements, state);
    initRemoteLink(elements, state);

    // Check backend status after initialization
    checkBackendStatus(elements, state);

    page.classList.add('is-ready');
  }

  function sanitizeCards(cards, templateMetrics) {
    if (!Array.isArray(cards) || !cards.length) {
      return [];
    }

    return cards.map((card, index) => {
      const id = typeof card.id === 'string' && card.id.trim().length
        ? card.id.trim()
        : buildObjectId(index + 1);

      const title = typeof card.title === 'string' && card.title.trim().length
        ? card.title.trim()
        : 'Object ' + String(index + 1).padStart(2, '0');

      const subtitle = typeof card.subtitle === 'string' && card.subtitle.trim().length
        ? card.subtitle.trim()
        : 'Identifier • Pending';

      const status = normalizeStatus(card.status);

      const metricsSource = Array.isArray(card.metrics) && card.metrics.length
        ? card.metrics
        : templateMetrics.length ? templateMetrics : buildDefaultMetrics();

      const metrics = metricsSource.slice(0, MAX_FEATURES).map((metric, metricIndex) => {
        const label = metric && typeof metric.label === 'string' && metric.label.trim().length
          ? metric.label.trim()
          : 'Feature ' + String(metricIndex + 1).padStart(2, '0');
        const value = metric && typeof metric.value === 'string' && metric.value.trim().length
          ? metric.value.trim()
          : 'Awaiting value';
        return { label, value };
      });

      return {
        id,
        title,
        subtitle,
        status,
        metrics,
        notes: typeof card.notes === 'string' ? card.notes : ''
      };
    });
  }

  function normalizeStatus(status) {
    const allowed = ['Draft', 'Configured', 'Mapped', 'Imported'];
    if (typeof status === 'string' && allowed.includes(status)) {
      return status;
    }
    return 'Draft';
  }

  function extractTemplateMetrics(template) {
    const metrics = [];
    if (!template || !template.content) {
      return metrics;
    }

    const sample = template.content.firstElementChild;
    if (!sample) {
      return metrics;
    }

    sample.querySelectorAll('.data-card__metrics > div').forEach((node) => {
      const labelNode = node.querySelector('dt');
      const valueNode = node.querySelector('dd');
      metrics.push({
        label: labelNode ? labelNode.textContent.trim() : 'Feature',
        value: valueNode ? valueNode.textContent.trim() : 'Awaiting value'
      });
    });

    return metrics;
  }

  function extractCardsFromDom(grid, templateMetrics) {
    if (!grid) {
      return [];
    }

    const rawCards = [];
    grid.querySelectorAll('[data-object-card]').forEach((card, index) => {
      const id = card.getAttribute('data-object-id') || buildObjectId(index + 1);
      const title = card.querySelector('h3') ? card.querySelector('h3').textContent.trim() : 'Object ' + (index + 1);
      const subtitle = card.querySelector('.data-card__subtitle')
        ? card.querySelector('.data-card__subtitle').textContent.trim()
        : 'Identifier • Pending';
      const statusText = card.querySelector('.data-card__status')
        ? card.querySelector('.data-card__status').textContent.trim()
        : 'Draft';
      const metrics = [];
      card.querySelectorAll('.data-card__metrics > div').forEach((metricNode) => {
        const dt = metricNode.querySelector('dt');
        const dd = metricNode.querySelector('dd');
        metrics.push({
          label: dt ? dt.textContent.trim() : '',
          value: dd ? dd.textContent.trim() : ''
        });
      });
      rawCards.push({
        id,
        title,
        subtitle,
        status: statusText,
        metrics,
        notes: card.dataset.objectNotes || ''
      });
    });

    return sanitizeCards(rawCards, templateMetrics);
  }
  function renderCards(elements, state) {
    const grid = elements.objectGrid;
    if (!grid) {
      return;
    }

    const fragment = document.createDocumentFragment();

    state.cards.forEach((cardData) => {
      const cardElement = buildCardElement(cardData, elements, state);
      fragment.appendChild(cardElement);
      normalizeCardElement(cardElement);
      applyStatusClasses(cardElement, cardData.status);
    });

    grid.innerHTML = '';
    grid.appendChild(fragment);

    const hasCards = state.cards.length > 0;
    const selectedIdExists = hasCards && state.selectedId && cardExists(state.cards, state.selectedId);
    state.selectedId = selectedIdExists ? state.selectedId : hasCards ? state.cards[0].id : null;

    const runnerIdExists = hasCards && state.runnerSelectionId && cardExists(state.cards, state.runnerSelectionId);
    state.runnerSelectionId = runnerIdExists ? state.runnerSelectionId : state.selectedId;

    if (state.selectedId) {
      const selector = '[data-object-id="' + state.selectedId + '"]';
      const selectedElement = grid.querySelector(selector);
      if (selectedElement) {
        selectedElement.classList.add('is-selected');
        selectedElement.setAttribute('aria-pressed', 'true');
      }
    }

    if (state.runnerSelectionId) {
      const runnerSelector = '[data-object-id="' + state.runnerSelectionId + '"]';
      const runnerElement = grid.querySelector(runnerSelector);
      const label = runnerElement ? getCardTitle(runnerElement) : '';
      updateSelectionStatus(elements.selectionStatus, label);
    } else {
      updateSelectionStatus(elements.selectionStatus, '');
    }

    if (state.pendingIntroId) {
      const pendingSelector = '[data-object-id="' + state.pendingIntroId + '"]';
      const newCard = grid.querySelector(pendingSelector);
      if (newCard) {
        bumpCardIntro(newCard);
      }
      state.pendingIntroId = null;
    }
  }

  function buildCardElement(cardData, elements, state) {
    const templateRoot = elements.cardTemplate.content.firstElementChild;
    const card = templateRoot.cloneNode(true);

    card.setAttribute('data-object-id', cardData.id);
    card.dataset.objectNotes = cardData.notes || '';

    const titleNode = card.querySelector('[data-card-title]') || card.querySelector('h3');
    if (titleNode) {
      titleNode.textContent = cardData.title || 'Untitled object';
    }

    const subtitleNode = card.querySelector('[data-card-subtitle]') || card.querySelector('.data-card__subtitle');
    if (subtitleNode) {
      subtitleNode.textContent = cardData.subtitle || 'Identifier • Pending';
    }

    const statusNode = card.querySelector('[data-card-status]') || card.querySelector('.data-card__status');
    if (statusNode) {
      statusNode.textContent = cardData.status || 'Draft';
    }

    const metricsContainer = card.querySelector('[data-card-metrics]') || card.querySelector('.data-card__metrics');
    if (metricsContainer) {
      const metricTemplate = metricsContainer.firstElementChild ? metricsContainer.firstElementChild.cloneNode(true) : null;
      metricsContainer.innerHTML = '';

      const metrics = Array.isArray(cardData.metrics) && cardData.metrics.length
        ? cardData.metrics
        : state.templateMetrics.length ? state.templateMetrics : buildDefaultMetrics();

      metrics.slice(0, MAX_FEATURES).forEach((metric, index) => {
        const item = metricTemplate ? metricTemplate.cloneNode(true) : document.createElement('div');
        let labelNode = item.querySelector('dt');
        if (!labelNode) {
          labelNode = document.createElement('dt');
          item.appendChild(labelNode);
        }
        labelNode.textContent = metric.label || 'Feature ' + String(index + 1).padStart(2, '0');

        let valueNode = item.querySelector('dd');
        if (!valueNode) {
          valueNode = document.createElement('dd');
          item.appendChild(valueNode);
        }
        valueNode.textContent = metric.value || 'Awaiting value';

        metricsContainer.appendChild(item);
      });
    }

    return card;
  }

  function normalizeCardElement(card) {
    if (!card) {
      return;
    }

    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-pressed', 'false');

    const actionButtons = card.querySelectorAll('[data-card-action]');
    actionButtons.forEach((button) => {
      button.type = 'button';
    });
  }

  function bumpCardIntro(card) {
    if (!card) {
      return;
    }

    card.classList.add('is-new');
    const handleAnimation = () => {
      card.classList.remove('is-new');
      card.removeEventListener('animationend', handleAnimation);
    };
    card.addEventListener('animationend', handleAnimation);
  }

  function applyStatusClasses(card, status) {
    if (!card) {
      return;
    }

    card.classList.remove('is-configured', 'is-mapped');
    const statusNode = card.querySelector('.data-card__status');
    const normalized = normalizeStatus(status);

    if (normalized === 'Configured') {
      card.classList.add('is-configured');
    } else if (normalized === 'Mapped') {
      card.classList.add('is-mapped');
    }

    if (statusNode) {
      statusNode.textContent = normalized;
    }
  }

  function initAddObject(elements, state) {
    if (!elements.addButton) {
      return;
    }

    elements.addButton.addEventListener('click', () => {
      state.objectCounter += 1;
      const newId = buildObjectId(state.objectCounter);
      const suffix = newId.split('-')[1] || String(state.objectCounter).padStart(3, '0');

      const newCard = createCardDataFromTemplate(state, {
        id: newId,
        title: 'New object ' + suffix,
        subtitle: 'Identifier • Pending',
        status: 'Draft',
        notes: ''
      });

      state.cards.push(newCard);
      state.selectedId = newId;
      state.runnerSelectionId = newId;
      state.pendingIntroId = newId;

      renderCards(elements, state);
      persistState(state);

      pushLog(state, elements.outputLog, 'Added manual object shell: ' + newCard.title + '.');
    });
  }

  function createCardDataFromTemplate(state, overrides) {
    const baseMetrics = overrides.metrics && overrides.metrics.length
      ? overrides.metrics
      : state.templateMetrics.length ? state.templateMetrics : buildDefaultMetrics();

    const metrics = baseMetrics.slice(0, MAX_FEATURES).map((metric, index) => ({
      label: metric && metric.label ? metric.label : 'Feature ' + String(index + 1).padStart(2, '0'),
      value: metric && metric.value ? metric.value : 'Awaiting value'
    }));

    return {
      id: overrides.id,
      title: overrides.title || 'New object',
      subtitle: overrides.subtitle || 'Identifier • Pending',
      status: normalizeStatus(overrides.status),
      metrics,
      notes: overrides.notes || ''
    };
  }

  function buildDefaultMetrics() {
    return [
      { label: 'Feature 01', value: 'Awaiting value' },
      { label: 'Feature 02', value: 'Awaiting value' },
      { label: 'Feature 03', value: 'Awaiting value' },
      { label: 'Feature 04', value: 'Awaiting value' }
    ];
  }

  function initImport(elements, state) {
    if (!elements.importInput) {
      return;
    }

    elements.importInput.addEventListener('change', (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) {
        return;
      }

      handleObjectImport(file, elements, state);
      event.target.value = '';
    });
  }

  function handleObjectImport(file, elements, state) {
    const reader = new FileReader();

    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : '';
      try {
        const parsed = parseImportedContent(file, text, state);
        state.objectCounter += 1;
        const newId = buildObjectId(state.objectCounter);
        const cardData = createCardDataFromTemplate(state, {
          id: newId,
          title: parsed.title,
          subtitle: parsed.subtitle,
          status: parsed.status,
          notes: parsed.notes,
          metrics: parsed.metrics
        });

        state.cards.push(cardData);
        state.selectedId = newId;
        state.runnerSelectionId = newId;
        state.pendingIntroId = newId;

        renderCards(elements, state);
        persistState(state);

        pushLog(state, elements.outputLog, 'Imported ' + file.name + ' into ' + cardData.title + '.');
      } catch (error) {
        pushLog(state, elements.outputLog, 'Import failed for ' + file.name + ': ' + (error.message || 'unrecognized format.'));
      }
    };

    reader.onerror = () => {
      pushLog(state, elements.outputLog, 'Could not read ' + file.name + '.');
    };

    reader.readAsText(file);
  }

  function parseImportedContent(file, text, state) {
    const extension = (file.name.split('.').pop() || '').toLowerCase();
    const fallbackTitle = file.name.replace(/\.[^.]+$/, '') || 'Imported object';

    if (extension === 'json' || looksLikeJson(text)) {
      return parseImportedJson(text, fallbackTitle, state, file.name);
    }

    if (extension === 'csv' || looksLikeCsv(text)) {
      return parseImportedCsv(text, fallbackTitle, state, file.name);
    }

    return parseImportedJson(text, fallbackTitle, state, file.name);
  }

  function parseImportedJson(text, fallbackTitle, state, fileName) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new Error('Invalid JSON');
    }

    const target = Array.isArray(data) ? data[0] : data;
    if (!target || typeof target !== 'object') {
      throw new Error('JSON must contain an object');
    }

    const entries = Object.entries(target)
      .filter(([, value]) => value !== null && value !== undefined && typeof value !== 'object');

    const metrics = mapEntriesToMetrics(entries, state);
    const title = typeof target.name === 'string' && target.name.trim().length
      ? target.name.trim()
      : typeof target.title === 'string' && target.title.trim().length
        ? target.title.trim()
        : fallbackTitle;

    const subtitle = typeof target.identifier === 'string' && target.identifier.trim().length
      ? target.identifier.trim()
      : typeof target.id === 'string' && target.id.trim().length
        ? target.id.trim()
        : 'Imported via JSON';

    return {
      title,
      subtitle,
      metrics,
      notes: 'Imported from ' + fileName,
      status: 'Imported'
    };
  }

  function parseImportedCsv(text, fallbackTitle, state, fileName) {
    const lines = text.split(/\r?\n/).map((line) => line.trim()).filter((line) => line.length);
    if (lines.length < 2) {
      throw new Error('CSV must contain a header row and at least one data row');
    }

    const headers = lines[0].split(',').map((value) => value.trim());
    const values = lines[1].split(',').map((value) => value.trim());

    const entries = headers.map((header, index) => [
      header || 'Column ' + (index + 1),
      values[index] || ''
    ]);

    const metrics = mapEntriesToMetrics(entries, state);

    return {
      title: fallbackTitle,
      subtitle: 'Imported via CSV',
      metrics,
      notes: 'Imported from ' + fileName,
      status: 'Imported'
    };
  }

  function mapEntriesToMetrics(entries, state) {
    if (!Array.isArray(entries) || !entries.length) {
      return cloneMetricArray(state.templateMetrics.length ? state.templateMetrics : buildDefaultMetrics());
    }

    return entries.slice(0, MAX_FEATURES).map(([label, rawValue], index) => ({
      label: typeof label === 'string' && label.trim().length ? label.trim() : 'Feature ' + String(index + 1).padStart(2, '0'),
      value: formatMetricValue(rawValue)
    }));
  }

  function formatMetricValue(value) {
    if (value === null || value === undefined) {
      return 'Awaiting value';
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }
    if (typeof value === 'string' && value.trim().length) {
      return value.trim();
    }
    return 'Awaiting value';
  }

  function looksLikeJson(text) {
    const trimmed = text.trim();
    return trimmed.startsWith('{') || trimmed.startsWith('[');
  }

  function looksLikeCsv(text) {
    return text.includes(',');
  }
  function initSelection(elements, state) {
    const grid = elements.objectGrid;
    if (!grid) {
      return;
    }

    grid.addEventListener('click', (event) => {
      const actionButton = event.target.closest('[data-card-action]');
      if (actionButton) {
        return;
      }

      const card = event.target.closest('[data-object-card]');
      if (!card) {
        return;
      }

      setSelectedCard(card, elements, state);
    });

    grid.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') {
        return;
      }

      const card = event.target.closest('[data-object-card]');
      if (!card) {
        return;
      }

      event.preventDefault();
      setSelectedCard(card, elements, state);
    });
  }

  function initChooseCard(elements, state) {
    if (!elements.chooseCardButton || !elements.objectLibrarySection) {
      return;
    }

    elements.chooseCardButton.addEventListener('click', () => {
      state.chooseMode = true;
      elements.objectLibrarySection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      elements.objectGrid.classList.add('is-choose-mode');
      updateSelectionStatus(elements.selectionStatus, '', { pendingSelection: true });
      pushLog(state, elements.outputLog, 'Choose from cards mode enabled.');
    });
  }

  function initCardActions(elements, state) {
    if (!elements.objectGrid) {
      return;
    }

    elements.objectGrid.addEventListener('click', (event) => {
      const actionButton = event.target.closest('[data-card-action]');
      if (!actionButton) {
        return;
      }

      const card = actionButton.closest('[data-object-card]');
      if (!card) {
        return;
      }

      const action = actionButton.getAttribute('data-card-action');
      const objectId = card.getAttribute('data-object-id');
      const wasChooseMode = state.chooseMode;

      if (action !== 'duplicate') {
        setSelectedCard(card, elements, state, { skipPersist: false });
      }

      if (action === 'open-shell') {
        openInspector(elements, state, card, 'open');
      } else if (action === 'configure') {
        openInspector(elements, state, card, 'configure');
      } else if (action === 'map-fields') {
        openInspector(elements, state, card, 'map');
      } else if (action === 'duplicate') {
        duplicateCard(card, elements, state);
      } else if (action === 'delete') {
        deleteCard(card, elements, state);
      }

      if (wasChooseMode && objectId) {
        const label = getCardTitle(card);
        pushLog(state, elements.outputLog, 'Single object selection set to ' + label + '.');
      }
    });
  }

  function duplicateCard(cardElement, elements, state) {
    if (!cardElement) {
      return;
    }

    const cardId = cardElement.getAttribute('data-object-id');
    const sourceIndex = state.cards.findIndex((card) => card.id === cardId);
    if (sourceIndex === -1) {
      return;
    }

    const sourceData = state.cards[sourceIndex];
    state.objectCounter += 1;
    const newId = buildObjectId(state.objectCounter);
    const sourceTitle = sourceData.title || sourceData.id;
    const cloneTitle = sourceTitle ? sourceTitle + ' copy' : 'Duplicated object';

    const cloneData = {
      id: newId,
      title: cloneTitle,
      subtitle: sourceData.subtitle,
      status: 'Draft',
      metrics: cloneMetricArray(sourceData.metrics),
      notes: ''
    };

    state.cards.splice(sourceIndex + 1, 0, cloneData);
    state.selectedId = newId;
    state.runnerSelectionId = newId;
    state.pendingIntroId = newId;

    renderCards(elements, state);
    persistState(state);

    pushLog(state, elements.outputLog, 'Duplicated layout from ' + sourceTitle + ' -> ' + cloneTitle + '.');
  }

  function cloneMetricArray(metrics) {
    const source = Array.isArray(metrics) && metrics.length ? metrics : buildDefaultMetrics();
    return source.slice(0, MAX_FEATURES).map((metric, index) => ({
      label: metric && metric.label ? metric.label : 'Feature ' + String(index + 1).padStart(2, '0'),
      value: metric && metric.value ? metric.value : 'Awaiting value'
    }));
  }

  function deleteCard(cardElement, elements, state) {
    if (!cardElement) {
      return;
    }

    const cardId = cardElement.getAttribute('data-object-id');
    const cardTitle = getCardTitle(cardElement) || cardId;

    // Don't allow deletion if it's the only card
    if (state.cards.length <= 1) {
      pushLog(state, elements.outputLog, 'Cannot delete the last remaining object.');
      return;
    }

    // Remove from state
    const cardIndex = state.cards.findIndex((card) => card.id === cardId);
    if (cardIndex !== -1) {
      state.cards.splice(cardIndex, 1);
    }

    // Update selections if this card was selected
    if (state.selectedId === cardId) {
      state.selectedId = state.cards.length > 0 ? state.cards[0].id : null;
    }
    if (state.runnerSelectionId === cardId) {
      state.runnerSelectionId = state.selectedId;
    }

    renderCards(elements, state);
    persistState(state);

    pushLog(state, elements.outputLog, 'Deleted object: ' + cardTitle + '.');
  }

  function initSingleUpload(elements, state) {
    if (!elements.singleUploadInput) {
      return;
    }

    elements.singleUploadInput.addEventListener('change', (event) => {
      const files = event.target.files ? Array.from(event.target.files) : [];
      state.uploads.single = files.length ? mapFile(files[0]) : null;
      renderUploadList(elements.singleUploadList, state.uploads.single ? [state.uploads.single] : [], {
        type: 'single',
        onRemove: () => {
          state.uploads.single = null;
          renderUploadList(elements.singleUploadList, [], { type: 'single' });
          pushLog(state, elements.outputLog, 'Cleared single object upload slot.');
        }
      });
      if (state.uploads.single) {
        pushLog(state, elements.outputLog, 'Single object upload ready: ' + state.uploads.single.name);
      }
      event.target.value = '';
    });
  }

  function initBulkUpload(elements, state) {
    if (!elements.bulkUploadInput) {
      return;
    }

    elements.bulkUploadInput.addEventListener('change', (event) => {
      const files = event.target.files ? Array.from(event.target.files) : [];
      if (!files.length) {
        return;
      }

      files.forEach((file) => {
        state.uploads.bulk.push(mapFile(file));
      });

      renderUploadList(elements.bulkUploadList, state.uploads.bulk, { type: 'bulk' });
      pushLog(state, elements.outputLog, 'Bulk dataset queue updated (' + state.uploads.bulk.length + ' file' + (state.uploads.bulk.length === 1 ? '' : 's') + ').');
      event.target.value = '';
    });
  }

  function initBulkListRemoval(elements, state) {
    const list = elements.bulkUploadList;
    if (!list) {
      return;
    }

    list.addEventListener('click', (event) => {
      const removeButton = event.target.closest('[data-remove-index]');
      if (!removeButton) {
        return;
      }

      const index = parseInt(removeButton.getAttribute('data-remove-index'), 10);
      if (Number.isNaN(index)) {
        return;
      }

      state.uploads.bulk.splice(index, 1);
      renderUploadList(elements.bulkUploadList, state.uploads.bulk, { type: 'bulk' });
      pushLog(state, elements.outputLog, 'Removed dataset file from queue. Remaining: ' + state.uploads.bulk.length + '.');
    });
  }

  function initBulkDownload(elements, state) {
    if (!elements.bulkDownloadButton) {
      return;
    }

    elements.bulkDownloadButton.addEventListener('click', async () => {
      if (!state.bulkResults || state.uploads.bulk.length === 0) {
        pushLog(state, elements.outputLog, 'No bulk results available for download.');
        return;
      }

      try {
        elements.bulkDownloadButton.disabled = true;
        elements.bulkDownloadButton.textContent = 'Preparing download...';

        // Use the first bulk file for processing (simple implementation)
        const firstBulkFile = state.uploads.bulk[0];
        if (!firstBulkFile) {
          throw new Error('No bulk file available');
        }

        pushLog(state, elements.outputLog, `Generating CSV download for ${firstBulkFile.name}...`);

        // Request CSV download from the API
        const result = await exoScanAPI.predictBulkFile(firstBulkFile.file, true, true); // download = true

        if (result.success) {
          // Create download link
          const url = window.URL.createObjectURL(result.data);
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          a.download = result.filename || 'bulk_predictions.csv';
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);

          pushLog(state, elements.outputLog, `✓ Download started: ${a.download}`);
        } else {
          throw new Error(result.error);
        }
      } catch (error) {
        pushLog(state, elements.outputLog, `✗ Download failed: ${error.message}`);
      } finally {
        elements.bulkDownloadButton.disabled = false;
        elements.bulkDownloadButton.textContent = 'Download Results';
      }
    });
  }

  function initRemoteLink(elements, state) {
    if (!elements.remoteLinkButton || !elements.outputLog) {
      return;
    }

    elements.remoteLinkButton.addEventListener('click', async () => {
      elements.remoteLinkButton.disabled = true;
      elements.remoteLinkButton.textContent = 'Running Prediction...';
      
      try {
        // Check if there's a selected card
        if (!state.selectedId) {
          throw new Error('No object selected. Please select an object from the library first.');
        }

        pushLog(state, elements.outputLog, 'Running prediction on selected object...');
        
        // Extract data from selected card
        const apiData = extractCardDataForAPI(state.selectedId, state);
        if (!apiData) {
          throw new Error('Selected object has no valid data for prediction');
        }

        // Check if all required fields are present
        const requiredFields = [
          "orbital_period", "stellar_radius", "rate_of_ascension", "declination",
          "transit_duration", "transit_depth", "planet_radius", "planet_temperature",
          "insolation_flux", "stellar_temperature"
        ];
        
        const missingFields = requiredFields.filter(field => apiData[field] === null || apiData[field] === undefined);
        if (missingFields.length > 0) {
          throw new Error(`Missing required data: ${missingFields.join(', ')}`);
        }

        // Run prediction
        const result = await exoScanAPI.predictSingle(apiData);
        
        if (result.success && result.data) {
          const cardTitle = state.cards.find(card => card.id === state.selectedId)?.title || state.selectedId;
          pushLog(state, elements.outputLog, `✓ Prediction completed for ${cardTitle}`);
          pushLog(state, elements.outputLog, `Prediction result: ${JSON.stringify(result.data)}`);
          
          // Display the prediction results
          displayPredictionResults([{ success: true, data: result.data }], elements, state);
        } else {
          throw new Error(result.error || 'Prediction failed');
        }
        
      } catch (error) {
        pushLog(state, elements.outputLog, 'Prediction failed: ' + error.message);
        console.error('Prediction error:', error);
      } finally {
        elements.remoteLinkButton.disabled = false;
        elements.remoteLinkButton.textContent = 'Run Prediction';
      }
    });
  }

  function initRunAction(elements, state) {
    if (!elements.runButton) {
      return;
    }

    elements.runButton.addEventListener('click', async () => {
      // Provide immediate user feedback
      elements.runButton.disabled = true;
      
      // Show loading animation
      const buttonText = elements.runButton.querySelector('.button-text');
      const loadingSpinner = elements.runButton.querySelector('.loading-spinner');
      if (buttonText) buttonText.style.display = 'none';
      if (loadingSpinner) loadingSpinner.style.display = 'flex';
      
      try {
        const summary = [];
        let predictionResults = [];

        // Determine what data to process
        let dataToProcess = [];
        
        if (state.uploads.single) {
          // Use uploaded single file data
          summary.push('Processing single JSON: ' + state.uploads.single.name);
          const fileResult = await exoScanAPI.processFileUpload(state.uploads.single.file, 'single');
          
          if (fileResult.success) {
            dataToProcess.push(fileResult.data);
          } else {
            throw new Error('Failed to process single file: ' + fileResult.error);
          }
        } else if (state.runnerSelectionId) {
          // Use selected card data
          const apiData = extractCardDataForAPI(state.runnerSelectionId, state);
          if (apiData) {
            const cardTitle = state.cards.find(card => card.id === state.runnerSelectionId)?.title || state.runnerSelectionId;
            summary.push('Processing selected object: ' + cardTitle);
            dataToProcess.push(apiData);
          } else {
            throw new Error('Selected object not found or has no valid data');
          }
        }

        // Process bulk uploads
        if (state.uploads.bulk.length > 0) {
          summary.push(`Processing ${state.uploads.bulk.length} bulk file(s)`);
          
          for (const bulkFile of state.uploads.bulk) {
            pushLog(state, elements.outputLog, `Processing bulk file: ${bulkFile.name}`);
            
            // Use the new bulk API endpoint
            const fileResult = await exoScanAPI.predictBulkFile(bulkFile.file, true); // has_raw_features = true
            
            if (fileResult.success) {
              // Extract predictions from the bulk result
              const predictions = fileResult.data.data.predictions || [];
              const statistics = fileResult.data.data.statistics || {};
              
              pushLog(state, elements.outputLog, `✓ Bulk file processed: ${predictions.length} predictions generated`);
              pushLog(state, elements.outputLog, `Statistics: ${JSON.stringify(statistics)}`);
              
              // Store bulk results for download
              state.bulkResults = fileResult.data;
              
              // Enable download button
              if (elements.bulkDownloadButton) {
                elements.bulkDownloadButton.disabled = false;
              }
              
              // Add all predictions to results
              predictions.forEach(pred => {
                predictionResults.push({ success: true, data: pred });
              });
            } else {
              pushLog(state, elements.outputLog, `✗ Failed to process ${bulkFile.name}: ${fileResult.error}`);
            }
          }
        }

        if (dataToProcess.length === 0 && predictionResults.length === 0) {
          throw new Error('No valid data to process. Please select an object or upload files.');
        }

        pushLog(state, elements.outputLog, summary.join('. ') + '. Starting predictions...');
        
        // Debug: Log the data being sent to the API
        if (dataToProcess.length === 1) {
          pushLog(state, elements.outputLog, 'Debug: Sending data to API: ' + JSON.stringify(dataToProcess[0], null, 2));
        }

        // Make API calls for predictions (only for single objects, bulk files are already processed)
        if (dataToProcess.length === 1 && predictionResults.length === 0) {
          // Single prediction
          const result = await exoScanAPI.predictSingle(dataToProcess[0]);
          if (result.success) {
            predictionResults.push({ success: true, data: result.data });
            pushLog(state, elements.outputLog, '✓ Single prediction completed successfully');
          } else {
            throw new Error('Prediction failed: ' + result.error);
          }
        } else if (dataToProcess.length > 1 && predictionResults.length === 0) {
          // Batch predictions for individual objects (not bulk files)
          pushLog(state, elements.outputLog, `Processing ${dataToProcess.length} objects...`);
          
          for (let i = 0; i < dataToProcess.length; i++) {
            try {
              const result = await exoScanAPI.predictSingle(dataToProcess[i]);
              if (result.success) {
                predictionResults.push({ success: true, data: result.data });
              } else {
                predictionResults.push({ success: false, error: result.error });
              }
            } catch (error) {
              predictionResults.push({ success: false, error: error.message });
            }
          }
          
          const successCount = predictionResults.filter(r => r.success).length;
          pushLog(state, elements.outputLog, `Batch processing completed: ${successCount}/${dataToProcess.length} successful`);
        }

        // Display results
        displayPredictionResults(predictionResults, elements, state);
        
        // Optional: Generate SHAP analysis for the first successful prediction
        const firstSuccess = predictionResults.find(r => r.success);
        if (firstSuccess && predictionResults.length === 1) {
          pushLog(state, elements.outputLog, 'Generating SHAP explanation...');
          try {
            const rawData = dataToProcess[0];
            const shapResult = await exoScanAPI.generateSHAPAnalysis(rawData);
            if (shapResult.success) {
              displaySHAPResults(shapResult.data, elements, state);
              pushLog(state, elements.outputLog, 'SHAP analysis completed');
            } else {
              pushLog(state, elements.outputLog, 'SHAP analysis not available: ' + (shapResult.error || 'Feature not implemented'));
            }
          } catch (shapError) {
            pushLog(state, elements.outputLog, 'SHAP analysis failed: ' + shapError.message);
          }
        }

      } catch (error) {
        pushLog(state, elements.outputLog, 'Error: ' + error.message);
        console.error('Prediction error:', error);
      } finally {
        // Reset button
        elements.runButton.disabled = false;
        
        // Hide loading animation
        const buttonText = elements.runButton.querySelector('.button-text');
        const loadingSpinner = elements.runButton.querySelector('.loading-spinner');
        if (buttonText) buttonText.style.display = 'block';
        if (loadingSpinner) loadingSpinner.style.display = 'none';
      }
    });
  }

  function initInspector(elements, state) {
    if (!elements.inspector || !elements.inspectorForm) {
      return;
    }

    const closeTriggers = elements.inspector.querySelectorAll('[data-action="close-inspector"]');
    closeTriggers.forEach((trigger) => {
      trigger.addEventListener('click', () => {
        hideInspector(elements, state);
      });
    });

    if (elements.inspectorBackdrop) {
      elements.inspectorBackdrop.addEventListener('click', () => {
        hideInspector(elements, state);
      });
    }

    elements.inspectorForm.addEventListener('submit', (event) => {
      event.preventDefault();
      applyInspectorChanges(new FormData(elements.inspectorForm), elements, state);
    });
  }

  function openInspector(elements, state, card, mode) {
    if (!elements.inspector || !elements.inspectorForm) {
      return;
    }

    const snapshot = createCardSnapshot(card);
    state.inspector.activeCard = card;
    state.inspector.snapshot = snapshot;
    state.inspector.mode = mode;
    state.inspector.lastFocused = document.activeElement && typeof document.activeElement.focus === 'function'
      ? document.activeElement
      : null;

    populateInspector(elements, snapshot, mode);
    showInspector(elements, state);

    const copy = getInspectorCopy(mode);
    const label = snapshot.title || snapshot.id || 'object';
    pushLog(state, elements.outputLog, copy.openLog + label + '.');

    if (elements.inspectorNameInput) {
      window.requestAnimationFrame(() => {
        elements.inspectorNameInput.focus();
        elements.inspectorNameInput.select();
      });
    }
  }

  function populateInspector(elements, snapshot, mode) {
    const copy = getInspectorCopy(mode);

    if (elements.inspectorModeLabel) {
      elements.inspectorModeLabel.textContent = copy.eyebrow;
    }

    if (elements.inspectorTitle) {
      elements.inspectorTitle.textContent = snapshot.title || snapshot.id || 'Object shell';
    }

    if (elements.inspectorMeta) {
      const statusText = snapshot.status ? ' • Status: ' + snapshot.status : '';
      elements.inspectorMeta.textContent = 'ID: ' + (snapshot.id || '—') + statusText;
    }

    if (elements.inspectorHelper) {
      elements.inspectorHelper.textContent = copy.helper;
    }

    if (elements.inspectorSaveButton) {
      elements.inspectorSaveButton.textContent = copy.saveLabel;
    }

    if (elements.inspectorForm) {
      elements.inspectorForm.dataset.mode = mode;
    }

    if (elements.inspectorNameInput) {
      elements.inspectorNameInput.value = snapshot.title || '';
      elements.inspectorNameInput.placeholder = mode === 'map'
        ? 'Enter object name or mapping label'
        : 'Enter object name';
    }

    if (elements.inspectorSubtitleInput) {
      elements.inspectorSubtitleInput.value = snapshot.subtitle || '';
    }

    if (elements.inspectorNotes) {
      elements.inspectorNotes.value = snapshot.notes || '';
    }

    if (elements.inspectorFeatures) {
      elements.inspectorFeatures.innerHTML = '';
      if (!snapshot.metrics.length) {
        const placeholder = document.createElement('p');
        placeholder.className = 'object-inspector__helper';
        placeholder.textContent = 'No feature slots yet — the final schema will add them.';
        elements.inspectorFeatures.appendChild(placeholder);
      } else {
        snapshot.metrics.forEach((metric, index) => {
          const field = document.createElement('label');
          field.className = 'object-inspector__field';

          const labelNode = document.createElement('span');
          labelNode.textContent = metric.label || 'Feature ' + String(index + 1).padStart(2, '0');
          field.appendChild(labelNode);

          const input = document.createElement('input');
          input.type = 'text';
          input.name = 'feature-value-' + index;
          input.dataset.featureIndex = String(index);
          input.value = metric.value || '';
          input.placeholder = mode === 'map' ? 'Mapping target or note' : 'Enter value';
          field.appendChild(input);

          elements.inspectorFeatures.appendChild(field);
        });
      }
    }
  }

  function showInspector(elements, state) {
    if (!elements.inspector || !elements.inspectorPanel) {
      return;
    }

    elements.inspector.hidden = false;
    elements.inspector.setAttribute('aria-hidden', 'false');
    elements.inspectorPanel.scrollTop = 0;
    window.requestAnimationFrame(() => {
      elements.inspector.classList.add('is-visible');
    });
    document.body.classList.add('is-inspector-open');

    if (state.inspector.keydownHandler) {
      document.removeEventListener('keydown', state.inspector.keydownHandler);
    }

    state.inspector.keydownHandler = (event) => {
      if (event.key === 'Escape') {
        hideInspector(elements, state);
      }
    };

    document.addEventListener('keydown', state.inspector.keydownHandler);
  }

  function hideInspector(elements, state) {
    if (!elements.inspector) {
      return;
    }

    elements.inspector.classList.remove('is-visible');
    elements.inspector.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('is-inspector-open');

    const panel = elements.inspectorPanel;
    if (panel) {
      const handleTransitionEnd = () => {
        elements.inspector.hidden = true;
        panel.removeEventListener('transitionend', handleTransitionEnd);
      };
      panel.addEventListener('transitionend', handleTransitionEnd, { once: true });
      window.setTimeout(() => {
        if (!elements.inspector.classList.contains('is-visible')) {
          elements.inspector.hidden = true;
        }
      }, 380);
    } else {
      elements.inspector.hidden = true;
    }

    if (state.inspector.keydownHandler) {
      document.removeEventListener('keydown', state.inspector.keydownHandler);
      state.inspector.keydownHandler = null;
    }

    if (elements.inspectorForm) {
      elements.inspectorForm.reset();
    }

    if (elements.inspectorFeatures) {
      elements.inspectorFeatures.innerHTML = '';
    }

    const refocusTarget = state.inspector.lastFocused;
    state.inspector.activeCard = null;
    state.inspector.snapshot = null;
    state.inspector.mode = 'open';
    state.inspector.lastFocused = null;

    if (refocusTarget && typeof refocusTarget.focus === 'function') {
      window.requestAnimationFrame(() => {
        refocusTarget.focus();
      });
    }
  }

  function createCardSnapshot(card) {
    const metrics = [];
    card.querySelectorAll('.data-card__metrics > div').forEach((metricNode) => {
      const labelNode = metricNode.querySelector('dt');
      const valueNode = metricNode.querySelector('dd');
      metrics.push({
        label: labelNode ? labelNode.textContent.trim() : '',
        value: valueNode ? valueNode.textContent.trim() : ''
      });
    });

    return {
      id: card.getAttribute('data-object-id') || '',
      title: card.querySelector('h3') ? card.querySelector('h3').textContent.trim() : '',
      subtitle: card.querySelector('.data-card__subtitle') ? card.querySelector('.data-card__subtitle').textContent.trim() : '',
      status: card.querySelector('.data-card__status') ? card.querySelector('.data-card__status').textContent.trim() : 'Draft',
      metrics,
      notes: card.dataset.objectNotes || ''
    };
  }

  function getInspectorCopy(mode) {
    const copies = {
      open: {
        eyebrow: 'Object shell',
        helper: 'Adjust placeholder values or descriptions. These updates stay on the frontend shell for now.',
        saveLabel: 'Save shell',
        openLog: 'Opened object shell for ',
        saveLog: 'Saved shell updates for '
      },
      configure: {
        eyebrow: 'Configuration',
        helper: 'Prep this object before handing it off to the backend. Update identifiers, feature values, and leave notes.',
        saveLabel: 'Apply configuration',
        openLog: 'Started configuration for ',
        saveLog: 'Captured configuration edits for '
      },
      map: {
        eyebrow: 'Field mapping',
        helper: 'Match each feature to the target schema or leave mapping hints. Nothing gets persisted yet.',
        saveLabel: 'Save mapping',
        openLog: 'Opening field mapping for ',
        saveLog: 'Captured mapping notes for '
      }
    };

    return copies[mode] || copies.open;
  }

  function applyInspectorChanges(formData, elements, state) {
    const card = state.inspector.activeCard;
    if (!card) {
      hideInspector(elements, state);
      return;
    }

    const snapshot = state.inspector.snapshot || createCardSnapshot(card);
    const cardData = findCardData(state, snapshot.id);
    let hasChanges = false;

    const titleValue = (formData.get('object-title') || '').trim();
    const nextTitle = titleValue || snapshot.title || 'Untitled object';
    const titleElement = card.querySelector('h3');
    if (titleElement && nextTitle !== snapshot.title) {
      titleElement.textContent = nextTitle;
      hasChanges = true;
    }
    if (cardData) {
      cardData.title = nextTitle;
    }

    const subtitleValue = (formData.get('object-subtitle') || '').trim();
    const nextSubtitle = subtitleValue || snapshot.subtitle || 'Identifier • Pending';
    const subtitleElement = card.querySelector('.data-card__subtitle');
    if (subtitleElement && nextSubtitle !== snapshot.subtitle) {
      subtitleElement.textContent = nextSubtitle;
      hasChanges = true;
    }
    if (cardData) {
      cardData.subtitle = nextSubtitle;
    }

    const notesValue = (formData.get('object-notes') || '').trim();
    if (notesValue !== (card.dataset.objectNotes || '')) {
      card.dataset.objectNotes = notesValue;
      hasChanges = true;
    }
    if (cardData) {
      cardData.notes = notesValue;
    }

    const metricNodes = card.querySelectorAll('.data-card__metrics > div');
    metricNodes.forEach((metricNode, index) => {
      const valueField = formData.get('feature-value-' + index);
      const trimmed = valueField != null ? String(valueField).trim() : '';
      const snapshotValue = snapshot.metrics[index] ? snapshot.metrics[index].value : '';
      const nextValue = trimmed || snapshotValue || 'Awaiting value';
      const dd = metricNode.querySelector('dd');
      if (dd && nextValue !== snapshotValue) {
        dd.textContent = nextValue;
        hasChanges = true;
      }
      if (cardData && cardData.metrics[index]) {
        cardData.metrics[index].value = nextValue;
      }
    });

    const label = snapshot.title || snapshot.id || 'object';
    const shouldForceStatusUpdate = !hasChanges && (state.inspector.mode === 'configure' || state.inspector.mode === 'map');

    let nextStatus = snapshot.status || 'Draft';
    if (hasChanges || shouldForceStatusUpdate) {
      if (state.inspector.mode === 'configure') {
        nextStatus = 'Configured';
      } else if (state.inspector.mode === 'map') {
        nextStatus = 'Mapped';
      } else {
        nextStatus = 'Draft';
      }
    }

    if (cardData) {
      cardData.status = nextStatus;
    }

    applyStatusClasses(card, nextStatus);

    setSelectedCard(card, elements, state, { skipPersist: true });

    if (hasChanges || shouldForceStatusUpdate) {
      persistState(state);
      const logMessage = hasChanges
        ? buildInspectorSaveLog(state.inspector.mode, card)
        : 'Marked ' + label + ' as ' + (state.inspector.mode === 'map' ? 'Mapped' : 'Configured') + '.';
      pushLog(state, elements.outputLog, logMessage);
    } else {
      pushLog(state, elements.outputLog, 'No changes captured for ' + label + '.');
    }

    hideInspector(elements, state);
  }

  function buildInspectorSaveLog(mode, card) {
    const info = getInspectorCopy(mode);
    const labelElement = card.querySelector('h3');
    const label = labelElement ? labelElement.textContent.trim() : card.getAttribute('data-object-id') || 'object';
    const base = info.saveLog || 'Saved changes for ';
    return base + label + '.';
  }
  function setSelectedCard(card, elements, state, options = {}) {
    if (!card) {
      return;
    }

    const previous = elements.objectGrid.querySelector('[data-object-card].is-selected');
    if (previous && previous !== card) {
      previous.classList.remove('is-selected');
      previous.setAttribute('aria-pressed', 'false');
    }

    card.classList.add('is-selected');
    card.setAttribute('aria-pressed', 'true');

    const objectId = card.getAttribute('data-object-id') || '';
    state.selectedId = objectId;

    if (!options.skipRunnerUpdate) {
      state.runnerSelectionId = objectId;
    }

    if (state.chooseMode) {
      state.chooseMode = false;
      elements.objectGrid.classList.remove('is-choose-mode');
    }

    if (!options.silent) {
      const label = getCardTitle(card);
      if (!options.skipRunnerUpdate) {
        updateSelectionStatus(elements.selectionStatus, label);
      } else if (state.runnerSelectionId) {
        const runnerSelector = '[data-object-id="' + state.runnerSelectionId + '"]';
        const runnerElement = elements.objectGrid.querySelector(runnerSelector);
        const runnerLabel = runnerElement ? getCardTitle(runnerElement) : '';
        updateSelectionStatus(elements.selectionStatus, runnerLabel);
      }
    }

    if (!options.skipPersist) {
      persistState(state);
    }
  }

  function getCardTitle(card) {
    const heading = card.querySelector('h3');
    return heading ? heading.textContent.trim() : card.getAttribute('data-object-id') || '';
  }

  function updateSelectionStatus(displayNode, label, options = {}) {
    if (!displayNode) {
      return;
    }

    if (options.pendingSelection) {
      displayNode.textContent = 'Single object: choose a card from the library.';
      return;
    }

    if (!label) {
      displayNode.textContent = 'Single object: no selection yet.';
      return;
    }

    displayNode.textContent = 'Single object ready: ' + label;
  }

  function findCardData(state, cardId) {
    return state.cards.find((card) => card.id === cardId) || null;
  }

  function mapFile(file) {
    return {
      name: file.name,
      size: file.size
    };
  }

  function renderUploadList(listNode, files, options) {
    if (!listNode) {
      return;
    }

    listNode.innerHTML = '';
    if (!files.length) {
      const empty = document.createElement('li');
      empty.className = 'upload-chip upload-chip--empty';
      empty.textContent = options && options.type === 'single' ? 'No file uploaded yet.' : 'Queue empty.';
      listNode.appendChild(empty);
      return;
    }

    files.forEach((file, index) => {
      const item = document.createElement('li');
      item.className = 'upload-chip';

      const label = document.createElement('span');
      label.className = 'upload-chip__label';
      label.textContent = file.name;
      item.appendChild(label);

      if (file.size != null) {
        const meta = document.createElement('span');
        meta.className = 'upload-chip__meta';
        meta.textContent = formatBytes(file.size);
        item.appendChild(meta);
      }

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'upload-chip__remove';
      remove.setAttribute('aria-label', 'Remove ' + file.name);
      remove.textContent = '×';

      if (options && options.type === 'single' && options.onRemove) {
        remove.addEventListener('click', options.onRemove);
      } else {
        remove.dataset.removeIndex = String(index);
      }

      item.appendChild(remove);
      listNode.appendChild(item);
    });
  }

  function formatBytes(bytes) {
    if (typeof bytes !== 'number' || Number.isNaN(bytes)) {
      return '';
    }

    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }

    const precision = value < 10 && unitIndex > 0 ? 1 : 0;
    return value.toFixed(precision) + ' ' + units[unitIndex];
  }

  function buildObjectId(counter) {
    return 'object-' + String(counter).padStart(3, '0');
  }

  function cardExists(cards, id) {
    return cards.some((card) => card.id === id);
  }

  function getHighestCounterFromCards(cards) {
    return cards.reduce((highest, card) => {
      const match = card.id && card.id.match(/(\d+)/);
      if (match) {
        const numeric = parseInt(match[1], 10);
        if (Number.isFinite(numeric) && numeric > highest) {
          return numeric;
        }
      }
      return highest;
    }, cards.length);
  }

  function loadStoredState() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (error) {
      return null;
    }
  }

  function persistState(state) {
    if (!state.storageEnabled) {
      return;
    }

    const payload = {
      cards: state.cards,
      selectedId: state.selectedId,
      runnerSelectionId: state.runnerSelectionId
    };

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      // ignore storage errors silently
    }
  }

  function pushLog(state, container, message) {
    if (!container || !message) {
      return;
    }

    const time = new Date();
    const entry = {
      message: message,
      timestamp: time
    };

    if (state && Array.isArray(state.logs)) {
      state.logs.unshift(entry);
      state.logs = state.logs.slice(0, 6);
      renderLogs(container, state.logs);
    } else {
      renderLogs(container, [entry]);
    }
  }

  function renderLogs(container, logs) {
    container.innerHTML = '';
    logs.forEach((entry) => {
      const row = document.createElement('p');
      row.className = 'prediction-log-entry';
      const time = entry.timestamp ? entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
      row.textContent = '[' + time + '] ' + entry.message;
      container.appendChild(row);
    });
  }

  // Helper function to extract data from selected card and convert to API format
  function extractCardDataForAPI(cardId, state) {
    const card = state.cards.find(c => c.id === cardId);
    if (!card || !card.metrics) {
      return null;
    }

    // Map the card metrics to the required API format
    const apiData = {
      "orbital_period": null,
      "stellar_radius": null,
      "rate_of_ascension": null,
      "declination": null,
      "transit_duration": null,
      "transit_depth": null,
      "planet_radius": null,
      "planet_temperature": null,
      "insolation_flux": null,
      "stellar_temperature": null
    };

    // Map card metrics to API fields based on the labels
    card.metrics.forEach(metric => {
      const label = metric.label.toLowerCase();
      const value = metric.value;

      // Convert value to number if possible, otherwise keep as null
      let numericValue = null;
      if (value && value !== 'Awaiting value' && value !== 'Pending validation' && value !== 'Awaiting calculation' && value !== 'Awaiting measurement' && value !== 'Awaiting analysis') {
        const parsed = parseFloat(value);
        if (!isNaN(parsed)) {
          numericValue = parsed;
        }
      }

      // Map based on label matching
      if (label.includes('orbital') && label.includes('period')) {
        apiData.orbital_period = numericValue;
      } else if (label.includes('stellar') && label.includes('radius')) {
        apiData.stellar_radius = numericValue;
      } else if (label.includes('rate') && label.includes('ascension')) {
        apiData.rate_of_ascension = numericValue;
      } else if (label.includes('declination')) {
        apiData.declination = numericValue;
      } else if (label.includes('transit') && label.includes('duration')) {
        apiData.transit_duration = numericValue;
      } else if (label.includes('transit') && label.includes('depth')) {
        apiData.transit_depth = numericValue;
      } else if (label.includes('planet') && label.includes('radius')) {
        apiData.planet_radius = numericValue;
      } else if (label.includes('planet') && label.includes('temperature')) {
        apiData.planet_temperature = numericValue;
      } else if (label.includes('insolation') && label.includes('flux')) {
        apiData.insolation_flux = numericValue;
      } else if (label.includes('stellar') && label.includes('temperature')) {
        apiData.stellar_temperature = numericValue;
      }
    });

    return apiData;
  }

  // Helper functions for API integration
  function displayPredictionResults(results, elements, state) {
    if (!results || results.length === 0) {
      pushLog(state, elements.outputLog, 'No prediction results to display');
      return;
    }

    const successfulResults = results.filter(r => r.success);
    const failedResults = results.filter(r => !r.success);

    if (successfulResults.length > 0) {
      pushLog(state, elements.outputLog, `Prediction Results (${successfulResults.length} successful):`);
      
      // Display the first successful result in the UI
      const firstResult = successfulResults[0];
      if (firstResult.data) {
        displayPredictionCard(firstResult.data, elements);
      }
      
      successfulResults.forEach((result, index) => {
        if (result.data) {
          // Log the result for debugging
          pushLog(state, elements.outputLog, `  ${index + 1}. ${result.data.predicted_label} (${(result.data.confidence * 100).toFixed(1)}% confidence)`);
        }
      });
    }

    if (failedResults.length > 0) {
      pushLog(state, elements.outputLog, `Failed predictions: ${failedResults.length}`);
      failedResults.forEach((result, index) => {
        pushLog(state, elements.outputLog, `  ${index + 1}. Error: ${result.error || 'Unknown error'}`);
      });
    }
  }

  // Display prediction results in the UI card
  function displayPredictionCard(predictionData, elements) {
    const resultsPanel = document.getElementById('prediction-results-panel');
    if (!resultsPanel) return;

    // Show the results panel
    resultsPanel.style.display = 'block';

    // Update prediction label
    const predictionLabel = resultsPanel.querySelector('.prediction-label');
    if (predictionLabel) {
      predictionLabel.textContent = predictionData.predicted_label || 'Exoplanet Classification';
    }

    // Update confidence score
    const confidenceValue = resultsPanel.querySelector('.confidence-value');
    if (confidenceValue && predictionData.confidence) {
      confidenceValue.textContent = `${(predictionData.confidence * 100).toFixed(1)}%`;
    }

    // Update probability bars
    if (predictionData.probabilities) {
      const probabilities = predictionData.probabilities;
      
      // Exoplanet probability
      const exoplanetBar = resultsPanel.querySelector('.exoplanet-bar');
      const exoplanetValue = resultsPanel.querySelector('.exoplanet-bar').parentElement.querySelector('.probability-value');
      if (exoplanetBar && probabilities.Exoplanet !== undefined) {
        const percentage = (probabilities.Exoplanet * 100).toFixed(1);
        exoplanetBar.style.width = `${percentage}%`;
        exoplanetValue.textContent = `${percentage}%`;
      }

      // Uncertain probability
      const uncertainBar = resultsPanel.querySelector('.uncertain-bar');
      const uncertainValue = resultsPanel.querySelector('.uncertain-bar').parentElement.querySelector('.probability-value');
      if (uncertainBar && probabilities.Uncertain !== undefined) {
        const percentage = (probabilities.Uncertain * 100).toFixed(1);
        uncertainBar.style.width = `${percentage}%`;
        uncertainValue.textContent = `${percentage}%`;
      }

      // Not Exoplanet probability
      const notExoplanetBar = resultsPanel.querySelector('.not-exoplanet-bar');
      const notExoplanetValue = resultsPanel.querySelector('.not-exoplanet-bar').parentElement.querySelector('.probability-value');
      if (notExoplanetBar && probabilities['Not Exoplanet'] !== undefined) {
        const percentage = (probabilities['Not Exoplanet'] * 100).toFixed(1);
        notExoplanetBar.style.width = `${percentage}%`;
        notExoplanetValue.textContent = `${percentage}%`;
      }
    }

    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function displaySHAPResults(shapData, elements, state) {
    const shapPanel = document.getElementById('shap-results-panel');
    if (!shapPanel) return;

    // Show the SHAP panel
    shapPanel.style.display = 'block';

    // Update SHAP summary
    const shapSummary = document.getElementById('shap-summary');
    if (shapSummary) {
      if (shapData.success) {
        shapSummary.innerHTML = `
          <div style="text-align: center;">
            <h4 style="color: var(--color-primary); margin-bottom: 8px;">SHAP Analysis Complete</h4>
            <p style="color: var(--color-white); margin: 0;">Feature importance analysis generated successfully</p>
          </div>
        `;
      } else {
        shapSummary.innerHTML = `
          <div style="text-align: center;">
            <p style="color: var(--color-warning); margin: 0;">${shapData.error || 'SHAP analysis not available'}</p>
          </div>
        `;
      }
    }

    // Display feature importance if available
    if (shapData.success && shapData.feature_importance) {
      displayFeatureImportance(shapData, elements);
    }

    // Log to console for debugging
    if (shapData.feature_importance && Array.isArray(shapData.feature_importance)) {
      pushLog(state, elements.outputLog, 'SHAP feature importance analysis completed');
    } else {
      pushLog(state, elements.outputLog, 'SHAP feature importance data not available');
    }

    // Display SHAP visualization plot if available
    if (shapData.plots && shapData.plots.summary_plot) {
      displaySHAPVisualization(shapData.plots.summary_plot, elements);
      pushLog(state, elements.outputLog, 'SHAP visualization generated and displayed');
      console.log('SHAP Summary Plot (base64):', shapData.plots.summary_plot);
    }

    // Scroll to SHAP results
    shapPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Display feature importance
  function displayFeatureImportance(shapData, elements) {
    const importanceList = document.getElementById('importance-list');
    if (!importanceList || !shapData.feature_names || !shapData.feature_importance) return;

    // Clear existing content
    importanceList.innerHTML = '';

    // Create feature importance items
    const features = shapData.feature_names;
    const importance = shapData.feature_importance;

    // Create array of feature-importance pairs and sort by importance
    const featureImportancePairs = features.map((feature, index) => ({
      feature: feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      importance: importance[index]
    })).sort((a, b) => b.importance - a.importance);

    // Display top features
    featureImportancePairs.slice(0, 10).forEach((item, index) => {
      const importanceItem = document.createElement('div');
      importanceItem.className = 'importance-item';
      importanceItem.innerHTML = `
        <span class="feature-name">${item.feature}</span>
        <span class="importance-value">${(item.importance * 100).toFixed(1)}%</span>
      `;
      importanceList.appendChild(importanceItem);
    });
  }

  // Display SHAP visualization plot
  function displaySHAPVisualization(base64Plot, elements) {
    // Find or create a container for the SHAP plot
    let shapPlotContainer = document.getElementById('shap-plot-container');
    
    if (!shapPlotContainer) {
      // Create the container if it doesn't exist
      shapPlotContainer = document.createElement('div');
      shapPlotContainer.id = 'shap-plot-container';
      shapPlotContainer.className = 'shap-plot-container';
      
      // Add a title
      const plotTitle = document.createElement('h4');
      plotTitle.textContent = 'SHAP Feature Importance Visualization';
      plotTitle.style.color = 'var(--color-primary)';
      plotTitle.style.marginBottom = '12px';
      shapPlotContainer.appendChild(plotTitle);
      
      // Insert the container after the feature importance section
      const featureImportanceSection = document.getElementById('feature-importance');
      if (featureImportanceSection && featureImportanceSection.parentNode) {
        featureImportanceSection.parentNode.insertBefore(shapPlotContainer, featureImportanceSection.nextSibling);
      }
    }

    // Remove any existing plot image
    const existingImg = shapPlotContainer.querySelector('.shap-plot-image');
    if (existingImg) {
      existingImg.remove();
    }

    // Create and add the new plot image
    const plotImage = document.createElement('img');
    plotImage.className = 'shap-plot-image';
    plotImage.src = `data:image/png;base64,${base64Plot}`;
    plotImage.alt = 'SHAP Feature Importance Plot';
    plotImage.style.cssText = `
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      background: var(--color-white);
      padding: 16px;
      margin-top: 12px;
    `;

    // Add error handling for image loading
    plotImage.onerror = function() {
      console.error('Failed to load SHAP plot image');
      this.alt = 'Error loading SHAP plot visualization';
      this.style.display = 'none';
      
      // Show an error message instead
      const errorMsg = document.createElement('div');
      errorMsg.className = 'shap-plot-error';
      errorMsg.style.cssText = `
        padding: 16px;
        background: var(--color-dark-gray);
        border-radius: 8px;
        color: var(--color-warning);
        text-align: center;
        margin-top: 12px;
      `;
      errorMsg.textContent = 'Error displaying SHAP visualization';
      shapPlotContainer.appendChild(errorMsg);
    };

    plotImage.onload = function() {
      console.log('SHAP plot visualization loaded successfully');
    };

    shapPlotContainer.appendChild(plotImage);
  }

  // Check backend status on page load
  async function checkBackendStatus(elements, state) {
    try {
      const status = await exoScanAPI.getBackendStatus();
      
      if (status.success && status.online) {
        pushLog(state, elements.outputLog, '✓ Backend connected and ready');
        
        // Display available capabilities
        const capabilities = Object.entries(status.capabilities)
          .filter(([_, available]) => available)
          .map(([name, _]) => name)
          .join(', ');
        
        if (capabilities) {
          pushLog(state, elements.outputLog, `Available features: ${capabilities}`);
        }
      } else {
        pushLog(state, elements.outputLog, 'Backend offline - running in demo mode');
        pushLog(state, elements.outputLog, 'Error: ' + (status.error || 'Connection failed'));
      }
    } catch (error) {
      pushLog(state, elements.outputLog, 'Could not check backend status - running in demo mode');
      console.warn('Backend status check failed:', error);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready, { once: true });
  } else {
    ready();
  }
})();
