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
        this.cameraMode = document.getElementById('cameraMode');
        this.videoMode = document.getElementById('videoMode');
        
        // 檔案上傳
        this.videoFile = document.getElementById('videoFile');
        this.fileName = document.getElementById('fileName');
        this.uploadBtn = document.getElementById('uploadBtn');
    }
    
    bindEvents() {
        // 深蹲按鈕
        this.squatBtn.addEventListener('click', () => this.showModeModal());
        
        // 模式選擇
        this.cameraMode.addEventListener('click', () => this.startCameraMode());
        this.videoMode.addEventListener('click', () => this.showVideoUploadModal());
        
        // 播放控制
        this.playBtn.addEventListener('click', () => this.resumePlayback());
        this.pauseBtn.addEventListener('click', () => this.pausePlayback());
        this.stopBtn.addEventListener('click', () => this.stopDetection());
        
        // 檔案上傳
        this.videoFile.addEventListener('change', (e) => this.handleFileSelect(e));
        this.uploadBtn.addEventListener('click', () => this.startVideoMode());
        
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
    
    showVideoUploadModal() {
        this.modeModal.style.display = 'none';
        this.videoUploadModal.style.display = 'block';
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
        
        // 這裡簡化處理，實際應用中需要上傳檔案到伺服器
        // 目前使用檔案路徑（需要用戶手動輸入完整路徑）
        const videoPath = prompt('請輸入影片檔案的完整路徑:');
        if (!videoPath) return;
        
        try {
            const response = await fetch('/start_detection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    mode: 'video',
                    video_path: videoPath
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
}

// 初始化應用程式
document.addEventListener('DOMContentLoaded', () => {
    new SquatDetectionApp();
});
