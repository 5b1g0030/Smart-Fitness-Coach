class SquatDetectionApp {
    constructor() {
        this.isDetecting = false;
        this.isPaused = false;
        this.statsInterval = null;
        
        this.initElements();
        this.bindEvents();
    }
    
    initElements() {
        // 主要元素
        this.squatBtn = document.getElementById('squatBtn');
        this.videoPlaceholder = document.getElementById('videoPlaceholder');
        this.videoFeed = document.getElementById('videoFeed');
        this.playbackControls = document.getElementById('playbackControls');
        
        // 控制按鈕
        this.playBtn = document.getElementById('playBtn');
        this.pauseBtn = document.getElementById('pauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        
        // 統計元素
        this.totalCount = document.getElementById('totalCount');
        this.correctCount = document.getElementById('correctCount');
        this.accuracy = document.getElementById('accuracy');
        
        // 模態視窗
        this.modeModal = document.getElementById('modeModal');
        this.videoUploadModal = document.getElementById('videoUploadModal');
        this.accuracyModal = document.getElementById('accuracyModal');
        this.cameraMode = document.getElementById('cameraMode');
        this.videoMode = document.getElementById('videoMode');
        this.accuracyMode = document.getElementById('accuracyMode');
        
        // 檔案上傳
        this.videoFile = document.getElementById('videoFile');
        this.fileName = document.getElementById('fileName');
        this.uploadBtn = document.getElementById('uploadBtn');
        
        // 準確度計算相關元素
        this.accuracyVideoFile = document.getElementById('accuracyVideoFile');
        this.accuracyFileName = document.getElementById('accuracyFileName');
        this.calculateBtn = document.getElementById('calculateBtn');
        this.progressArea = document.getElementById('progressArea');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.resultArea = document.getElementById('resultArea');
        this.avgAccuracy = document.getElementById('avgAccuracy');
        this.totalFrames = document.getElementById('totalFrames');
        this.validFrames = document.getElementById('validFrames');
        this.closeResultBtn = document.getElementById('closeResultBtn');
    }
    
    bindEvents() {
        // 深蹲按鈕
        this.squatBtn.addEventListener('click', () => this.showModeModal());
        
        // 模式選擇
        this.cameraMode.addEventListener('click', () => this.startCameraMode());
        this.videoMode.addEventListener('click', () => this.showVideoUploadModal());
        this.accuracyMode.addEventListener('click', () => this.showAccuracyModal());
        
        // 播放控制
        this.playBtn.addEventListener('click', () => this.resumePlayback());
        this.pauseBtn.addEventListener('click', () => this.pausePlayback());
        this.stopBtn.addEventListener('click', () => this.stopDetection());
        
        // 檔案上傳
        this.videoFile.addEventListener('change', (e) => this.handleFileSelect(e));
        this.uploadBtn.addEventListener('click', () => this.startVideoMode());
        
        // 準確度計算
        this.accuracyVideoFile.addEventListener('change', (e) => this.handleAccuracyFileSelect(e));
        this.calculateBtn.addEventListener('click', () => this.startAccuracyCalculation());
        this.closeResultBtn.addEventListener('click', () => this.closeAccuracyModal());
        
        // 模態視窗關閉
        document.querySelectorAll('.close').forEach(closeBtn => {
            closeBtn.addEventListener('click', (e) => {
                e.target.closest('.modal').style.display = 'none';
            });
        });
        
        // 點擊外部關閉模態視窗
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.style.display = 'none';
            }
        });
    }
    
    showModeModal() {
        this.modeModal.style.display = 'block';
    }
    
    // 影片上傳
    showVideoUploadModal() {
        this.modeModal.style.display = 'none';          // 隱藏視窗
        this.videoUploadModal.style.display = 'block';  // 顯示影片上傳視窗
    }
    
    async startCameraMode() {
        this.modeModal.style.display = 'none';
        
        try {
            const response = await fetch('/start_detection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ mode: 'camera' })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.startDetection();
            } else {
                alert(`啟動攝像頭失敗: ${result.message}`);
            }
        } catch (error) {
            alert(`發生錯誤: ${error.message}`);
        }
    }
    
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.fileName.textContent = `已選擇: ${file.name}`;
            this.fileName.style.display = 'block';
            this.uploadBtn.style.display = 'inline-block';
        }
    }
    
    async startVideoMode() {
        const file = this.videoFile.files[0];
        if (!file) {
            alert('請選擇影片檔案');
            return;
        }
        
        this.videoUploadModal.style.display = 'none';
        
        // 上傳檔案到伺服器
        const formData = new FormData();
        formData.append('video', file);
        
        try {
            // 先上傳檔案
            const uploadResponse = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });
            
            const uploadResult = await uploadResponse.json();
            
            if (!uploadResult.success) {
                alert(`上傳失敗: ${uploadResult.message}`);
                return;
            }
            
            // 使用伺服器返回的檔案路徑開始檢測
            const response = await fetch('/start_detection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    mode: 'video',
                    video_path: uploadResult.file_path  // 使用伺服器路徑
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.startDetection();
            } else {
                alert(`載入影片失敗: ${result.message}`);
            }
        } catch (error) {
            alert(`發生錯誤: ${error.message}`);
        }
    }
    
    startDetection() {
        this.isDetecting = true;
        this.isPaused = false;
        
        // 隱藏佔位符，顯示視訊
        this.videoPlaceholder.style.display = 'none';
        this.videoFeed.style.display = 'block';
        this.videoFeed.src = '/video_feed?' + new Date().getTime();
        
        // 顯示控制按鈕
        this.playbackControls.style.display = 'flex';
        this.updateControlButtons();
        
        // 開始更新統計
        this.startStatsUpdate();
    }
    
    async pausePlayback() {
        this.isPaused = true;
        this.updateControlButtons();
        
        try {
            await fetch('/control_playback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ action: 'pause' })
            });
        } catch (error) {
            console.error('暫停失敗:', error);
        }
    }
    
    async resumePlayback() {
        this.isPaused = false;
        this.updateControlButtons();
        
        try {
            await fetch('/control_playback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ action: 'resume' })
            });
        } catch (error) {
            console.error('繼續播放失敗:', error);
        }
    }
    
    async stopDetection() {
        this.isDetecting = false;
        this.isPaused = false;
        
        try {
            await fetch('/control_playback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ action: 'stop' })
            });
        } catch (error) {
            console.error('停止失敗:', error);
        }
        
        // 恢復初始狀態
        this.videoFeed.style.display = 'none';
        this.videoFeed.src = '';
        this.videoPlaceholder.style.display = 'flex';
        this.playbackControls.style.display = 'none';
        
        // 停止統計更新
        this.stopStatsUpdate();
        
        // 重置檔案選擇
        this.videoFile.value = '';
        this.fileName.style.display = 'none';
        this.uploadBtn.style.display = 'none';
    }
    
    updateControlButtons() {
        if (this.isPaused) {
            this.playBtn.style.display = 'inline-block';
            this.pauseBtn.style.display = 'none';
        } else {
            this.playBtn.style.display = 'none';
            this.pauseBtn.style.display = 'inline-block';
        }
    }
    
    startStatsUpdate() {
        this.statsInterval = setInterval(async () => {
            try {
                const response = await fetch('/get_stats');
                const stats = await response.json();
                this.updateStats(stats);
            } catch (error) {
                console.error('獲取統計失敗:', error);
            }
        }, 1000);
    }
    
    stopStatsUpdate() {
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }
    }
    
    updateStats(stats) {
        this.totalCount.textContent = stats.squat_count;
        this.correctCount.textContent = stats.correct_squat_count;
        this.accuracy.textContent = `${stats.accuracy.toFixed(1)}%`;
    }
    
    showAccuracyModal() {
        this.modeModal.style.display = 'none';
        this.accuracyModal.style.display = 'block';
        this.resetAccuracyModal();
    }
    
    closeAccuracyModal() {
        this.accuracyModal.style.display = 'none';
        this.resetAccuracyModal();
    }
    
    resetAccuracyModal() {
        this.accuracyVideoFile.value = '';
        this.accuracyFileName.style.display = 'none';
        this.calculateBtn.style.display = 'none';
        this.progressArea.style.display = 'none';
        this.resultArea.style.display = 'none';
        this.progressFill.style.width = '0%';
        this.progressText.textContent = '準備中...';
    }
    
    handleAccuracyFileSelect(event) {
        console.log('handleAccuracyFileSelect 被觸發'); // 除錯用
        console.log('event.target.files:', event.target.files); // 除錯用
        
        const file = event.target.files[0];
        if (file) {
            console.log('選擇的檔案:', file.name); // 除錯用
            this.accuracyFileName.textContent = `已選擇: ${file.name}`;
            this.accuracyFileName.style.display = 'block';
            this.calculateBtn.style.display = 'inline-block';
        } else {
            console.log('沒有選擇檔案'); // 除錯用
        }
    }
    
    async startAccuracyCalculation() {
        const file = this.accuracyVideoFile.files[0];
        if (!file) {
            alert('請選擇影片檔案');
            return;
        }
        
        // 隱藏按鈕，顯示進度條
        this.calculateBtn.style.display = 'none';
        this.progressArea.style.display = 'block';
        this.resultArea.style.display = 'none';
        
        try {
            // 上傳檔案
            this.updateProgress(5, '正在上傳影片...');
            const formData = new FormData();
            formData.append('video', file);
            
            const uploadResponse = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });
            
            const uploadResult = await uploadResponse.json();
            
            if (!uploadResult.success) {
                throw new Error(`上傳失敗: ${uploadResult.message}`);
            }
            
            // 開始計算準確度 - 使用 EventSource 來處理 SSE
            this.updateProgress(10, '開始分析影片...');
            
            // 創建 EventSource 來接收進度更新
            const eventSource = new EventSource(`/calculate_accuracy_stream?video_path=${encodeURIComponent(uploadResult.file_path)}`);
            
            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('收到數據:', data);  // 調試用
                    
                    if (data.type === 'progress') {
                        this.updateProgress(data.progress, data.message);
                    } else if (data.type === 'result') {
                        this.updateProgress(100, '分析完成！');
                        this.showAccuracyResult(data);
                        eventSource.close();
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    }
                } catch (parseError) {
                    console.error('解析數據失敗:', parseError);
                    eventSource.close();
                    throw new Error('數據解析失敗');
                }
            };
            
            eventSource.onerror = (error) => {
                console.error('EventSource 錯誤:', error);
                eventSource.close();
                throw new Error('連接中斷，請重試');
            };
            
            // 5分鐘超時保護
            setTimeout(() => {
                if (eventSource.readyState !== EventSource.CLOSED) {
                    eventSource.close();
                    throw new Error('分析超時，請檢查影片檔案大小');
                }
            }, 300000); // 5分鐘
            
        } catch (error) {
            console.error('計算準確度失敗:', error);
            this.updateProgress(0, `錯誤: ${error.message}`);
            setTimeout(() => {
                this.progressArea.style.display = 'none';
                this.calculateBtn.style.display = 'inline-block';
            }, 3000);
        }
    }
    
    updateProgress(percentage, message) {
        this.progressFill.style.width = `${percentage}%`;
        this.progressText.textContent = message;
    }
    
    showAccuracyResult(data) {
        setTimeout(() => {
            this.progressArea.style.display = 'none';
            this.resultArea.style.display = 'block';
            
            // 顯示主要準確度（使用深蹲幀的平均值，與播放模式更一致）
            const mainAccuracy = data.squat_accuracy || data.average_accuracy;
            this.avgAccuracy.textContent = `${mainAccuracy.toFixed(1)}%`;
            this.totalFrames.textContent = data.total_frames;
            this.validFrames.textContent = data.valid_frames;
            
            // 如果有詳細資訊，顯示額外的統計
            if (data.squat_count !== undefined) {
                // 創建詳細資訊區域
                let detailsHtml = `
                    <div class="accuracy-details" style="margin-top: 15px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
                        <h4>詳細分析結果</h4>
                        <div class="detail-item">
                            <span class="detail-label">所有幀平均準確度:</span>
                            <span class="detail-value">${data.average_accuracy.toFixed(1)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">深蹲動作準確度:</span>
                            <span class="detail-value">${(data.squat_accuracy || 0).toFixed(1)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">計數式準確率:</span>
                            <span class="detail-value">${(data.count_based_accuracy || 0).toFixed(1)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">檢測到深蹲次數:</span>
                            <span class="detail-value">${data.squat_count || 0}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">正確深蹲次數:</span>
                            <span class="detail-value">${data.correct_squat_count || 0}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">深蹲動作幀數:</span>
                            <span class="detail-value">${data.squat_frames || 0}</span>
                        </div>
                    </div>
                `;
                
                // 將詳細資訊插入結果區域
                const existingDetails = this.resultArea.querySelector('.accuracy-details');
                if (existingDetails) {
                    existingDetails.remove();
                }
                this.closeResultBtn.insertAdjacentHTML('beforebegin', detailsHtml);
            }
        }, 1000);
    }
}

// 初始化應用程式
document.addEventListener('DOMContentLoaded', () => {
    new SquatDetectionApp();
});
