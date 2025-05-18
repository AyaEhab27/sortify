class WasteClassifier {
    constructor(apiUrls) {
        this.apiUrls = Array.isArray(apiUrls) ? apiUrls : [apiUrls];
        this.cache = new Map(); 
    }

    async classifyImage(imageFile) {
        const cacheKey = await this.generateCacheKey(imageFile);
        

        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        const formData = new FormData();
        formData.append('file', imageFile);

        let lastError = null;

        for (const apiUrl of this.apiUrls) {
            try {
                const response = await fetch(`${apiUrl}/classify`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    throw new Error('Invalid response format from server');
                }

                const data = await response.json();
                
                if (!data.all_predictions || !Array.isArray(data.all_predictions)) {
                    throw new Error('Invalid response data structure');
                }

                this.cache.set(cacheKey, data);
                return data;
            } catch (error) {
                lastError = error;
                console.warn(`Failed with ${apiUrl}:`, error);
                continue;
            }
        }

        throw lastError || new Error('All API endpoints failed');
    }

    async generateCacheKey(file) {
        return `${file.name}-${file.size}-${file.lastModified}`;
    }
}

const classifier = new WasteClassifier([
    'https://sortify-15.onrender.com'
]);

export async function handleApiError(error) {
    let errorType = 'An unexpected error occurred';
    let details = error.message || 'Please try again or contact support.';
    
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        errorType = 'Network error';
        details = 'Please check your internet connection and try again.';
    } else if (error.message.includes('404')) {
        errorType = 'API endpoint not found';
        details = 'The classification service is currently unavailable.';
    } else if (error.message.includes('413')) {
        errorType = 'File too large';
        details = 'The image file is too large.';
    }
    
    return { error: errorType, details };
}

export default classifier;
