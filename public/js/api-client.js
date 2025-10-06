/**
 * ExoScan API Client
 * 
 * Handles all communication with the backend API for the prediction page.
 * Provides a clean interface for data preprocessing, model prediction, 
 * SHAP analysis, and other backend services.
 */

class ExoScanAPI {
  constructor(baseURL = '') {
    this.baseURL = baseURL;
    this.endpoints = {
      prediction: '/api/prediction/predictiondata',
      bulk: '/api/prediction/bulk',
      bulkCsv: '/api/prediction/bulk/csv',
      graph: '/api/prediction/graph', 
      fillExample: '/api/prediction/fillexample',
      shapReport: '/api/report/shap',
      status: '/api/prediction/status'
    };
  }

  /**
   * Generic fetch wrapper with error handling
   */
  async _fetch(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    };

    const config = { ...defaultOptions, ...options };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      } else {
        return await response.text();
      }
    } catch (error) {
      console.error(`API call failed for ${endpoint}:`, error);
      throw new Error(`API Error: ${error.message}`);
    }
  }

  /**
   * Preprocess and predict single exoplanet object
   */
  async predictSingle(exoplanetData) {
    try {
      const response = await this._fetch(this.endpoints.prediction, {
        method: 'POST',
        body: JSON.stringify(exoplanetData)
      });

      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Batch predict multiple exoplanet objects
   */
  async predictBatch(exoplanetDataArray) {
    try {
      const predictions = [];
      
      // Process in batches to avoid overwhelming the server
      const batchSize = 10;
      for (let i = 0; i < exoplanetDataArray.length; i += batchSize) {
        const batch = exoplanetDataArray.slice(i, i + batchSize);
        const batchPromises = batch.map(data => this.predictSingle(data));
        const batchResults = await Promise.all(batchPromises);
        predictions.push(...batchResults);
      }

      return {
        success: true,
        data: predictions,
        error: null,
        totalProcessed: predictions.length
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Get example data to fill prediction form
   */
  async getExampleData() {
    try {
      const response = await this._fetch(this.endpoints.fillExample, {
        method: 'GET'
      });

      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Get prediction visualization data
   */
  async getPredictionGraphData() {
    try {
      const response = await this._fetch(this.endpoints.graph, {
        method: 'GET'
      });

      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Generate SHAP explanation for prediction
   */
  async generateSHAPAnalysis(predictionData) {
    try {
      const response = await this._fetch(this.endpoints.shapReport, {
        method: 'POST',
        body: JSON.stringify(predictionData)
      });

      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Validate exoplanet data format before sending to backend
   */
  validateExoplanetData(data) {
    const errors = [];
    
    if (!data || typeof data !== 'object') {
      errors.push('Data must be a valid object');
      return { valid: false, errors };
    }

    // Check for required fields (flexible column naming supported by backend)
    const hasNumericData = Object.values(data).some(value => 
      typeof value === 'number' && !isNaN(value)
    );

    if (!hasNumericData) {
      errors.push('Data must contain at least one numeric measurement');
    }

    // Check for reasonable value ranges
    Object.entries(data).forEach(([key, value]) => {
      if (typeof value === 'number') {
        if (value < 0 && !['ra', 'dec', 'declination'].some(coord => key.toLowerCase().includes(coord))) {
          errors.push(`${key} should not be negative`);
        }
        if (Math.abs(value) > 1e10) {
          errors.push(`${key} value seems unreasonably large`);
        }
      }
    });

    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Bulk predict with file upload
   */
  async predictBulkFile(file, hasRawFeatures = true, downloadResults = false) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('has_raw_features', hasRawFeatures.toString());

      const endpoint = downloadResults ? this.endpoints.bulkCsv : this.endpoints.bulk;
      
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (downloadResults) {
        // For CSV download, return the blob
        const blob = await response.blob();
        return {
          success: true,
          data: blob,
          filename: `predictions_${file.name}`,
          error: null
        };
      } else {
        // For JSON response
        const result = await response.json();
        return result;
      }
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Bulk predict with JSON data
   */
  async predictBulkData(dataArray) {
    try {
      const response = await this._fetch(this.endpoints.bulk, {
        method: 'POST',
        body: JSON.stringify(dataArray)
      });

      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Get API status and model information
   */
  async getStatus() {
    try {
      const response = await this._fetch(this.endpoints.status, {
        method: 'GET'
      });

      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Process file upload for prediction
   */
  async processFileUpload(file, type = 'single') {
    return new Promise((resolve, reject) => {
      if (!file) {
        reject(new Error('No file provided'));
        return;
      }

      // Check file type
      const allowedTypes = ['application/json', 'text/csv', 'text/plain'];
      if (!allowedTypes.includes(file.type) && !file.name.match(/\.(json|csv|txt)$/i)) {
        reject(new Error('File must be JSON, CSV, or TXT format'));
        return;
      }

      const reader = new FileReader();
      
      reader.onload = async (e) => {
        try {
          const content = e.target.result;
          let data;

          if (file.type === 'application/json' || file.name.toLowerCase().endsWith('.json')) {
            // Parse JSON
            data = JSON.parse(content);
          } else if (file.type === 'text/csv' || file.name.toLowerCase().endsWith('.csv')) {
            // Parse CSV (simple implementation)
            data = this.parseCSV(content);
          } else {
            // Try to parse as JSON first, then as simple key-value pairs
            try {
              data = JSON.parse(content);
            } catch {
              data = this.parseKeyValue(content);
            }
          }

          // Validate the parsed data
          if (Array.isArray(data)) {
            // Batch data
            const validationResults = data.map(item => this.validateExoplanetData(item));
            const invalidItems = validationResults.filter(result => !result.valid);
            
            if (invalidItems.length > 0) {
              throw new Error(`${invalidItems.length} items have validation errors`);
            }
          } else {
            // Single object
            const validation = this.validateExoplanetData(data);
            if (!validation.valid) {
              throw new Error('Data validation failed: ' + validation.errors.join(', '));
            }
          }

          resolve({
            success: true,
            data: data,
            filename: file.name,
            type: type,
            error: null
          });

        } catch (error) {
          reject(new Error(`File processing failed: ${error.message}`));
        }
      };

      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };

      reader.readAsText(file);
    });
  }

  /**
   * Simple CSV parser
   */
  parseCSV(csvContent) {
    const lines = csvContent.trim().split('\n');
    if (lines.length < 2) {
      throw new Error('CSV must have header and at least one data row');
    }

    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const data = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
      const row = {};
      
      headers.forEach((header, index) => {
        const value = values[index];
        // Try to convert to number if possible
        const numValue = parseFloat(value);
        row[header] = isNaN(numValue) ? value : numValue;
      });
      
      data.push(row);
    }

    return data.length === 1 ? data[0] : data;
  }

  /**
   * Parse simple key-value text format
   */
  parseKeyValue(content) {
    const lines = content.trim().split('\n');
    const data = {};

    lines.forEach(line => {
      const [key, ...valueParts] = line.split(':');
      if (key && valueParts.length > 0) {
        const value = valueParts.join(':').trim();
        const numValue = parseFloat(value);
        data[key.trim()] = isNaN(numValue) ? value : numValue;
      }
    });

    return data;
  }

  /**
   * Health check endpoint
   */
  async healthCheck() {
    try {
      const response = await this._fetch('/health', {
        method: 'GET'
      });

      return {
        success: true,
        data: response,
        error: null
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error.message
      };
    }
  }

  /**
   * Get backend status and capabilities
   */
  async getBackendStatus() {
    try {
      // Try to get example data as a way to test backend connectivity
      const response = await this.getExampleData();
      
      return {
        success: true,
        online: true,
        capabilities: {
          prediction: true,
          preprocessing: true,
          shap: true,
          fileUpload: true
        },
        error: null
      };
    } catch (error) {
      return {
        success: false,
        online: false,
        capabilities: {},
        error: error.message
      };
    }
  }
}

// Create global instance
const exoScanAPI = new ExoScanAPI('http://localhost:8080');

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ExoScanAPI, exoScanAPI };
}

// Make available globally
window.ExoScanAPI = ExoScanAPI;
window.exoScanAPI = exoScanAPI;