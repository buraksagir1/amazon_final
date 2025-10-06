// services/streamService.js
import { StreamChat } from 'stream-chat';
import WebSocket from 'ws';
import speechToTextService from '../lib/speechToText.js';

class StreamTranscriptionService {
    constructor() {
        this.serverClient = StreamChat.getInstance(
            process.env.STREAM_API_KEY,
            process.env.STREAM_SECRET_KEY
        );
        this.activeConnections = new Map();
    }

    async startTranscriptionForCall(callId, userId) {
        try {
            if (this.activeConnections.has(callId)) {
                console.log(`Transcription already active for call: ${callId}`);
                return false;
            }

            const ws = speechToTextService.createRealtimeConnection();

            ws.on('open', () => {
                console.log(`Transcription started for call: ${callId}`);
                this.activeConnections.set(callId, {
                    ws,
                    userId,
                    startTime: Date.now()
                });
            });

            ws.on('message', async (data) => {
                try {
                    const result = JSON.parse(data);

                    if (result.message_type === 'FinalTranscript' && result.text) {
                        await this.sendSubtitleToCall(callId, {
                            text: result.text,
                            confidence: result.confidence,
                            userId: userId,
                            timestamp: new Date().toISOString()
                        });
                    }
                } catch (error) {
                    console.error('Message processing error:', error);
                }
            });

            ws.on('error', (error) => {
                console.error(`WebSocket error for call ${callId}:`, error);
                this.activeConnections.delete(callId);
            });

            ws.on('close', () => {
                console.log(`Transcription ended for call: ${callId}`);
                this.activeConnections.delete(callId);
            });

            return true;

        } catch (error) {
            console.error('Start transcription error:', error);
            throw error;
        }
    }

    stopTranscriptionForCall(callId) {
        const connection = this.activeConnections.get(callId);
        if (connection) {
            connection.ws.close();
            this.activeConnections.delete(callId);
            return true;
        }
        return false;
    }

    sendAudioData(callId, audioData) {
        const connection = this.activeConnections.get(callId);
        if (connection && connection.ws.readyState === WebSocket.OPEN) {
            connection.ws.send(JSON.stringify({
                audio_data: audioData
            }));
            return true;
        }
        return false;
    }

    async sendSubtitleToCall(callId, subtitleData) {
        try {
            const channel = this.serverClient.channel('call', callId);

            await channel.sendEvent({
                type: 'subtitle_received',
                user_id: subtitleData.userId,
                data: {
                    text: subtitleData.text,
                    confidence: subtitleData.confidence,
                    timestamp: subtitleData.timestamp
                }
            });

            return true;
        } catch (error) {
            console.error('Send subtitle error:', error);
            return false;
        }
    }

    getActiveConnections() {
        const connections = {};
        this.activeConnections.forEach((value, key) => {
            connections[key] = {
                userId: value.userId,
                startTime: value.startTime,
                duration: Date.now() - value.startTime
            };
        });
        return connections;
    }

    isTranscriptionActive(callId) {
        return this.activeConnections.has(callId);
    }

    cleanup() {
        this.activeConnections.forEach((connection, callId) => {
            connection.ws.close();
        });
        this.activeConnections.clear();
    }
}

export default new StreamTranscriptionService();