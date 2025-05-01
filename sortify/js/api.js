class TFJSModelLoader {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.model = null;
        this.isLoading = false;
    }

    async load() {
        if (this.model) return this.model;
        if (this.isLoading) {
            await new Promise(resolve => {
                const checkInterval = setInterval(() => {
                    if (!this.isLoading) {
                        clearInterval(checkInterval);
                        resolve();
                    }
                }, 100);
            });
            return this.model;
        }

        this.isLoading = true;
        try {
            this.model = await tf.loadGraphModel(this.modelPath);
            console.log('Model loaded successfully');
            this.isLoading = false;
            return this.model;
        } catch (error) {
            console.error('Error loading model:', error);
            this.isLoading = false;
            throw error;
        }
    }

    isReady() {
        return !!this.model && !this.isLoading;
    }
}