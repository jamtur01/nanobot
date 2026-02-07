/**
 * WhatsApp client wrapper using Baileys.
 * Based on OpenClaw's working implementation.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
import makeWASocket, {
  DisconnectReason,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
} from '@whiskeysockets/baileys';

import { Boom } from '@hapi/boom';
import qrcode from 'qrcode-terminal';
import pino from 'pino';
import { rmSync } from 'fs';

const VERSION = '0.1.0';

export interface InboundMessage {
  id: string;
  sender: string;
  content: string;
  timestamp: number;
  isGroup: boolean;
}

export interface WhatsAppClientOptions {
  authDir: string;
  onMessage: (msg: InboundMessage) => void;
  onQR: (qr: string) => void;
  onStatus: (status: string) => void;
}

export class WhatsAppClient {
  private sock: any = null;
  private options: WhatsAppClientOptions;
  private reconnecting = false;
  /** IDs of messages we sent, so we can skip our own outbound echoes */
  private sentMessageIds = new Set<string>();
  /** Cache of LID user → phone user for resolving opaque LIDs */
  private lidToPhone = new Map<string, string>();
  /** Auth state reference for accessing LID mappings */
  private authState: any = null;

  constructor(options: WhatsAppClientOptions) {
    this.options = options;
  }

  async connect(): Promise<void> {
    const logger = pino({ level: 'silent' });
    const { state, saveCreds } = await useMultiFileAuthState(this.options.authDir);
    this.authState = state;
    const { version } = await fetchLatestBaileysVersion();

    console.log(`Using Baileys version: ${version.join('.')}`);

    // Seed LID→phone mapping from our own credentials
    this._seedOwnLIDMapping(state.creds);

    // Create socket following OpenClaw's pattern
    this.sock = makeWASocket({
      auth: {
        creds: state.creds,
        keys: makeCacheableSignalKeyStore(state.keys, logger),
      },
      version,
      logger,
      printQRInTerminal: false,
      browser: ['nanobot', 'cli', VERSION],
      syncFullHistory: false,
      markOnlineOnConnect: false,
    });

    // Handle WebSocket errors
    if (this.sock.ws && typeof this.sock.ws.on === 'function') {
      this.sock.ws.on('error', (err: Error) => {
        console.error('WebSocket error:', err.message);
      });
    }

    // Handle connection updates
    this.sock.ev.on('connection.update', async (update: any) => {
      const { connection, lastDisconnect, qr } = update;

      if (qr) {
        // Display QR code in terminal for interactive login
        console.log('\n📱 Scan this QR code with WhatsApp (Linked Devices):\n');
        (qrcode as any).generate(qr, { small: true }, (code: string) => {
          console.log(code);
        });
        this.options.onQR(qr);
      }

      if (connection === 'close') {
        const statusCode = (lastDisconnect?.error as Boom)?.output?.statusCode;
        const isLoggedOut = statusCode === DisconnectReason.loggedOut;

        console.log(`Connection closed. Status: ${statusCode}, Logged out: ${isLoggedOut}`);
        this.options.onStatus('disconnected');

        if (!this.reconnecting) {
          this.reconnecting = true;

          if (isLoggedOut) {
            // Clear stale auth so Baileys generates a fresh QR on reconnect
            console.log('Session expired — clearing auth for fresh QR...');
            try { rmSync(this.options.authDir, { recursive: true, force: true }); } catch {}
          }

          const delay = isLoggedOut ? 1 : 5;
          console.log(`Reconnecting in ${delay}s...`);
          setTimeout(() => {
            this.reconnecting = false;
            this.connect();
          }, delay * 1000);
        }
      } else if (connection === 'open') {
        console.log('✅ Connected to WhatsApp');
        this.options.onStatus('connected');
      }
    });

    // Save credentials on update
    this.sock.ev.on('creds.update', saveCreds);

    // Handle incoming messages
    this.sock.ev.on('messages.upsert', async ({ messages, type }: { messages: any[]; type: string }) => {
      if (type !== 'notify') return;

      for (const msg of messages) {
        // Skip messages that *we* sent (prevents echo loops) while still
        // allowing the user's own "Message Yourself" chat to work.
        if (this.sentMessageIds.delete(msg.key.id || '')) continue;

        // Skip status updates
        if (msg.key.remoteJid === 'status@broadcast') continue;

        const content = this.extractMessageContent(msg);
        if (!content) continue;

        const isGroup = msg.key.remoteJid?.endsWith('@g.us') || false;

        // Resolve LID to phone-number JID so allowFrom lists work
        const rawJid: string = msg.key.remoteJid || '';
        const sender = this._resolveToPhoneJid(rawJid);

        this.options.onMessage({
          id: msg.key.id || '',
          sender,
          content,
          timestamp: msg.messageTimestamp as number,
          isGroup,
        });
      }
    });
  }

  /**
   * Seed the LID→phone cache from our own credentials.
   * creds.me has { id: "15037348571:24@s.whatsapp.net", lid: "194506284601577:24@lid" }
   */
  private _seedOwnLIDMapping(creds: any): void {
    const me = creds?.me;
    if (!me?.id || !me?.lid) return;

    const phoneUser = me.id.split(':')[0].split('@')[0];
    const lidUser = me.lid.split(':')[0].split('@')[0];

    if (phoneUser && lidUser) {
      this.lidToPhone.set(lidUser, phoneUser);
      console.log(`LID mapping: ${lidUser}@lid → ${phoneUser}@s.whatsapp.net`);
    }

    // Also try to load stored reverse mappings from the auth keys
    this._loadStoredMappings();
  }

  /**
   * Load LID→phone mappings stored by Baileys in the auth state.
   */
  private async _loadStoredMappings(): Promise<void> {
    try {
      const keys = this.authState?.keys;
      if (!keys?.get) return;

      // Baileys stores lid-mapping entries; reverse entries have _reverse suffix
      const stored = await keys.get('lid-mapping', []);
      if (stored && typeof stored === 'object') {
        for (const [key, value] of Object.entries(stored)) {
          if (key.endsWith('_reverse') && typeof value === 'string') {
            const lidUser = key.replace('_reverse', '');
            this.lidToPhone.set(lidUser, value);
          }
        }
      }
    } catch {
      // Silently ignore - mapping will fall back to raw JID
    }
  }

  /**
   * Resolve a JID, converting @lid JIDs to @s.whatsapp.net when possible.
   */
  private _resolveToPhoneJid(jid: string): string {
    if (!jid.endsWith('@lid')) return jid;

    const lidUser = jid.split(':')[0].split('@')[0];
    const phoneUser = this.lidToPhone.get(lidUser);

    if (phoneUser) {
      return `${phoneUser}@s.whatsapp.net`;
    }

    // Unknown LID — return as-is
    console.log(`Warning: unresolved LID ${jid}`);
    return jid;
  }

  private extractMessageContent(msg: any): string | null {
    const message = msg.message;
    if (!message) return null;

    // Text message
    if (message.conversation) {
      return message.conversation;
    }

    // Extended text (reply, link preview)
    if (message.extendedTextMessage?.text) {
      return message.extendedTextMessage.text;
    }

    // Image with caption
    if (message.imageMessage?.caption) {
      return `[Image] ${message.imageMessage.caption}`;
    }

    // Video with caption
    if (message.videoMessage?.caption) {
      return `[Video] ${message.videoMessage.caption}`;
    }

    // Document with caption
    if (message.documentMessage?.caption) {
      return `[Document] ${message.documentMessage.caption}`;
    }

    // Voice/Audio message
    if (message.audioMessage) {
      return `[Voice Message]`;
    }

    return null;
  }

  async sendMessage(to: string, text: string): Promise<void> {
    if (!this.sock) {
      throw new Error('Not connected');
    }

    const sent = await this.sock.sendMessage(to, { text });
    // Track the ID so we skip this message when it echoes back
    if (sent?.key?.id) {
      this.sentMessageIds.add(sent.key.id);
      // Safety: cap the set size to prevent unbounded growth
      if (this.sentMessageIds.size > 500) {
        const first = this.sentMessageIds.values().next().value;
        if (first !== undefined) this.sentMessageIds.delete(first);
      }
    }
  }

  async disconnect(): Promise<void> {
    if (this.sock) {
      this.sock.end(undefined);
      this.sock = null;
    }
  }
}
