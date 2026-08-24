import Foundation

/// The Web Portal's entire client: one self-contained HTML page (inline CSS/JS, no external
/// requests) served at `GET /`. Colors are pulled directly from the dark variants in
/// `Assets.xcassets` (`ThemeBackground`/`ThemeAccent`/etc.) so the web client matches the native
/// app's look rather than approximating it. Talks to the server exclusively over one WebSocket
/// connection (`/api/ws`) using small JSON messages — see `WebPortalManager` for the protocol.
nonisolated enum WebPortalAssets {
    static let indexHTML = """
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <title>DarkAI — Web Portal</title>
    <style>
    :root {
      --background: #020202;
      --card: rgba(18, 5, 5, 0.92);
      --accent: #FF0033;
      --accent-cyan: #FF5500;
      --accent-rose: #990011;
      --text-primary: #FFFFFF;
      --text-secondary: #CF9F9F;
      --text-muted: #9E7070;
      --border: #3A1212;
      --glow: rgba(255, 0, 51, 0.4);
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    html, body { height: 100%; margin: 0; overscroll-behavior: none; }
    body {
      background: var(--background);
      color: var(--text-primary);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
      display: flex;
      flex-direction: column;
    }
    #app { display: flex; height: 100dvh; width: 100%; }

    /* ---------- Login ---------- */
    #login {
      margin: auto;
      width: min(320px, 90vw);
      padding: 32px 28px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      text-align: center;
      box-shadow: 0 0 40px var(--glow);
    }
    #login h1 {
      font-size: 20px;
      letter-spacing: 0.08em;
      margin: 0 0 6px;
      background: linear-gradient(90deg, var(--accent), var(--accent-cyan));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    #login p { color: var(--text-secondary); font-size: 13px; margin: 0 0 22px; }
    #login input {
      width: 100%;
      background: var(--background);
      border: 1px solid var(--border);
      color: var(--text-primary);
      font-size: 22px;
      letter-spacing: 0.3em;
      text-align: center;
      border-radius: 12px;
      padding: 14px 10px;
      margin-bottom: 14px;
    }
    #login input:focus { outline: none; border-color: var(--accent); }
    #login button, .btn {
      width: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-rose));
      color: #fff;
      border: none;
      border-radius: 12px;
      padding: 13px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
    }
    #login button:active, .btn:active { opacity: 0.8; }
    #loginError { color: var(--accent); font-size: 12px; margin-top: 12px; min-height: 16px; }

    /* ---------- Chat shell ---------- */
    #chat { display: none; flex: 1; min-height: 0; }
    #chat.visible { display: flex; }

    #sidebar {
      width: 260px;
      flex: none;
      background: var(--card);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      transition: transform 0.25s ease;
    }
    #sidebar-header {
      padding: 18px 16px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
    }
    #sidebar-header h2 {
      font-size: 13px;
      letter-spacing: 0.15em;
      margin: 0;
      color: var(--text-primary);
    }
    #newChatBtn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 8px;
      width: 30px;
      height: 30px;
      font-size: 18px;
      line-height: 1;
      cursor: pointer;
    }
    #convList { flex: 1; overflow-y: auto; padding: 8px; }
    .convItem {
      padding: 12px 12px;
      border-radius: 10px;
      margin-bottom: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .convItem:hover { background: rgba(255,0,51,0.08); }
    .convItem.active { background: rgba(255,0,51,0.15); border: 1px solid var(--accent); }
    .convTitle {
      font-size: 13px;
      color: var(--text-primary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .convDelete {
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 15px;
      cursor: pointer;
      flex: none;
      padding: 2px 4px;
    }
    .convDelete:hover { color: var(--accent); }

    #main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
    #topbar {
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 12px;
    }
    #menuBtn {
      display: none;
      background: none;
      border: none;
      color: var(--text-primary);
      font-size: 20px;
      cursor: pointer;
    }
    #topbar h1 {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.1em;
      margin: 0;
      background: linear-gradient(90deg, var(--accent), var(--accent-cyan));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    #status { margin-left: auto; font-size: 11px; color: var(--text-muted); }

    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .msgRow { display: flex; }
    .msgRow.user { justify-content: flex-end; }
    .bubble {
      max-width: min(560px, 80%);
      padding: 12px 16px;
      border-radius: 16px;
      font-size: 14px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .msgRow.user .bubble {
      background: linear-gradient(135deg, var(--accent), var(--accent-rose));
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .msgRow.assistant .bubble {
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text-primary);
      border-bottom-left-radius: 4px;
    }
    .msgRow.assistant .bubble img {
      max-width: 100%;
      border-radius: 10px;
      margin-top: 8px;
      display: block;
    }
    .bubble.streaming::after {
      content: "▋";
      opacity: 0.6;
      animation: blink 1s step-start infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }

    #composer {
      border-top: 1px solid var(--border);
      padding: 14px 20px;
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }
    #textInput {
      flex: 1;
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text-primary);
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 14px;
      resize: none;
      max-height: 140px;
      font-family: inherit;
    }
    #textInput:focus { outline: none; border-color: var(--accent); }
    #sendBtn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 42px;
      height: 42px;
      flex: none;
      font-size: 16px;
      cursor: pointer;
    }
    #sendBtn:disabled { opacity: 0.4; }
    #empty { margin: auto; text-align: center; color: var(--text-muted); font-size: 13px; padding: 40px; }

    @media (max-width: 720px) {
      #menuBtn { display: block; }
      #sidebar {
        position: fixed;
        inset: 0 auto 0 0;
        z-index: 20;
        transform: translateX(-100%);
        box-shadow: 0 0 40px rgba(0,0,0,0.6);
      }
      #sidebar.open { transform: translateX(0); }
      #sidebarScrim {
        display: none;
        position: fixed; inset: 0;
        background: rgba(0,0,0,0.5);
        z-index: 19;
      }
      #sidebarScrim.visible { display: block; }
    }

    /* ---------- Settings ---------- */
    #settingsBtn {
      background: none;
      border: none;
      color: var(--text-primary);
      font-size: 17px;
      cursor: pointer;
      margin-left: 10px;
      flex: none;
    }
    #settingsPanel {
      display: none;
      position: fixed;
      inset: 0;
      background: var(--background);
      z-index: 30;
      flex-direction: column;
    }
    #settingsPanel.visible { display: flex; }
    #settingsHeader {
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 12px;
      flex: none;
    }
    #settingsHeader button {
      background: none;
      border: none;
      color: var(--text-primary);
      font-size: 20px;
      cursor: pointer;
    }
    #settingsHeader h1 {
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.12em;
      margin: 0;
      color: var(--text-secondary);
    }
    #settingsBody {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      max-width: 640px;
      width: 100%;
      margin: 0 auto;
    }
    #settingsNotice {
      display: none;
      font-size: 12px;
      padding: 10px 14px;
      border-radius: 10px;
      margin-bottom: 16px;
    }
    #settingsNotice.visible { display: block; }
    #settingsNotice.error { background: rgba(255,0,51,0.15); border: 1px solid var(--accent); color: var(--text-primary); }
    #settingsNotice.info { background: rgba(255,85,0,0.12); border: 1px solid var(--accent-cyan); color: var(--text-primary); }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .cardHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: var(--text-primary);
      margin-bottom: 10px;
    }
    .cardTextarea, .cardInput {
      width: 100%;
      background: var(--background);
      border: 1px solid var(--border);
      color: var(--text-primary);
      border-radius: 8px;
      padding: 10px;
      font-size: 13px;
      font-family: inherit;
    }
    .cardTextarea { resize: vertical; }
    .settingsSubtle { font-size: 11px; color: var(--text-muted); line-height: 1.5; }
    .badge {
      font-size: 10px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(153,0,17,0.3);
      color: var(--text-primary);
      flex: none;
    }
    .badge.small { font-size: 9px; padding: 2px 5px; flex: none; }
    .linkBtn {
      background: none;
      border: none;
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      padding: 0;
    }
    .dangerBtn {
      width: 100%;
      background: rgba(153,0,17,0.2);
      border: 1px solid rgba(153,0,17,0.5);
      color: var(--text-primary);
      border-radius: 8px;
      padding: 10px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      margin-top: 10px;
    }
    .settingsRowHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 10px 0 6px;
    }
    .settingsRow {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 0;
      border-top: 1px solid var(--border);
    }
    .settingsRow:first-of-type { border-top: none; }
    .settingsRow .convDelete { flex: none; }
    .modelRow {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px;
      background: var(--background);
      border-radius: 10px;
      margin-bottom: 8px;
    }
    .modelRow .smallBtn {
      background: var(--border);
      color: var(--text-primary);
      border: none;
      border-radius: 8px;
      padding: 8px 14px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      flex: none;
    }
    .modelRow .smallBtn:disabled { opacity: 0.5; cursor: default; }
    .loadBanner {
      display: none;
      margin-bottom: 10px;
      padding: 10px 12px;
      background: rgba(255,85,0,0.1);
      border: 1px solid var(--accent-cyan);
      border-radius: 10px;
      font-size: 12px;
      color: var(--text-primary);
    }
    .loadBanner.visible { display: block; }
    .loadBanner .loadBarTrack {
      margin-top: 8px;
      height: 6px;
      border-radius: 3px;
      background: var(--border);
      overflow: hidden;
    }
    .loadBanner .loadBarFill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-cyan));
      transition: width 0.2s ease;
    }
    .rangeRow { margin-bottom: 16px; }
    .rangeTop {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-secondary);
      margin-bottom: 6px;
    }
    .rangeValue { color: var(--accent); font-weight: 700; font-family: monospace; }
    .rangeRow input[type=range] { width: 100%; accent-color: var(--accent); }
    .toggleRow {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 12px;
      font-size: 13px;
      color: var(--text-secondary);
    }
    .switch { position: relative; display: inline-block; width: 44px; height: 26px; flex: none; }
    .switch input { opacity: 0; width: 0; height: 0; position: absolute; }
    .switch .slider {
      position: absolute; inset: 0; cursor: pointer;
      background: var(--border); transition: 0.2s; border-radius: 26px;
    }
    .switch .slider::before {
      content: ""; position: absolute; height: 20px; width: 20px; left: 3px; bottom: 3px;
      background: #fff; transition: 0.2s; border-radius: 50%;
    }
    .switch input:checked + .slider { background: var(--accent); }
    .switch input:checked + .slider::before { transform: translateX(18px); }
    </style>
    </head>
    <body>

    <div id="app">
      <div id="login">
        <h1>DARKAI WEB PORTAL</h1>
        <p>Enter the PIN shown in Settings → Web Portal on your phone.</p>
        <input id="pinInput" inputmode="numeric" maxlength="8" placeholder="••••••" autocomplete="off">
        <button class="btn" id="pinSubmit">Unlock</button>
        <div id="loginError"></div>
      </div>

      <div id="chat">
        <div id="sidebarScrim"></div>
        <div id="sidebar">
          <div id="sidebar-header">
            <h2>CHATS</h2>
            <button id="newChatBtn">+</button>
          </div>
          <div id="convList"></div>
        </div>
        <div id="main">
          <div id="topbar">
            <button id="menuBtn">☰</button>
            <h1>DARKAI</h1>
            <span id="status">connecting…</span>
            <button id="settingsBtn" title="Settings">⚙</button>
          </div>
          <div id="messages"><div id="empty">Select a chat, or start a new one.</div></div>
          <div id="composer">
            <textarea id="textInput" rows="1" placeholder="Message DarkAI…"></textarea>
            <button id="sendBtn">↑</button>
          </div>
        </div>
      </div>

      <div id="settingsPanel">
        <div id="settingsHeader">
          <button id="settingsClose">←</button>
          <h1>SETTINGS</h1>
        </div>
        <div id="settingsBody">
          <div id="settingsNotice"></div>

          <div class="card">
            <div class="cardHeader"><span>CUSTOM SYSTEM INSTRUCTIONS</span></div>
            <textarea id="setCustomInstructions" class="cardTextarea" rows="4"></textarea>
          </div>

          <div class="card">
            <div class="cardHeader"><span>MODELS</span></div>
            <div id="modelLoadBanner" class="loadBanner"></div>
            <div id="llmModelList"></div>
            <div id="coremlModelList"></div>
            <div class="settingsSubtle">Downloading or importing a model still needs the app on your phone — this only switches between models already installed there.</div>
          </div>

          <div class="card">
            <div class="cardHeader"><span>LLM SETTINGS</span></div>
            <div id="llmSettingsBody"></div>
          </div>

          <div class="card">
            <div class="cardHeader">
              <span>RAG</span>
              <label class="switch"><input type="checkbox" id="setEnableRAG"><span class="slider"></span></label>
            </div>
          </div>

          <div class="card">
            <div class="cardHeader">
              <span>CONVERSATIONAL MEMORIES</span>
              <label class="switch"><input type="checkbox" id="setEnableMemories"><span class="slider"></span></label>
            </div>
            <div id="memoriesBody"></div>
          </div>

          <div class="card">
            <div class="cardHeader">
              <span>INTERNET ACCESS</span>
              <label class="switch"><input type="checkbox" id="setInternetAccess"><span class="slider"></span></label>
            </div>
            <div id="internetBody">
              <input type="password" id="setBraveKey" class="cardInput" placeholder="Brave Search API key" autocomplete="off">
              <div id="braveKeyStatus" class="settingsSubtle" style="margin-top:8px;"></div>
            </div>
          </div>

          <div class="card">
            <div class="cardHeader"><span>PERSONALITY MATRIX</span><span id="personalityBadge" class="badge"></span></div>
            <div class="settingsSubtle">Gradually adapts tone to how you write. Resetting erases it completely, for every model.</div>
            <button class="dangerBtn" id="resetPersonalityBtn">Reset Personality</button>
          </div>

          <div class="card">
            <div class="cardHeader"><span>RESPONSE FEEDBACK</span><span id="feedbackCounts" class="settingsSubtle"></span></div>
            <div id="feedbackList"></div>
          </div>
        </div>
      </div>
    </div>

    <script>
    (function() {
      var ws = null;
      var conversations = [];
      var activeId = null;
      var streamingBubble = null;
      var reconnectDelay = 1000;
      var isModelLoading = false;
      var lastModels = {llm: [], coreml: []};

      var loginEl = document.getElementById('login');
      var chatEl = document.getElementById('chat');
      var pinInput = document.getElementById('pinInput');
      var pinSubmit = document.getElementById('pinSubmit');
      var loginError = document.getElementById('loginError');
      var convListEl = document.getElementById('convList');
      var messagesEl = document.getElementById('messages');
      var textInput = document.getElementById('textInput');
      var sendBtn = document.getElementById('sendBtn');
      var statusEl = document.getElementById('status');
      var newChatBtn = document.getElementById('newChatBtn');
      var menuBtn = document.getElementById('menuBtn');
      var sidebar = document.getElementById('sidebar');
      var sidebarScrim = document.getElementById('sidebarScrim');
      var settingsBtn = document.getElementById('settingsBtn');
      var settingsPanel = document.getElementById('settingsPanel');
      var settingsClose = document.getElementById('settingsClose');
      var settingsNotice = document.getElementById('settingsNotice');

      function connect() {
        var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(proto + '//' + location.host + '/api/ws');
        ws.onopen = function() {
          reconnectDelay = 1000;
        };
        ws.onmessage = function(e) {
          try { handle(JSON.parse(e.data)); } catch (err) { console.error(err); }
        };
        ws.onclose = function() {
          statusEl.textContent = 'reconnecting…';
          setTimeout(connect, reconnectDelay);
          reconnectDelay = Math.min(reconnectDelay * 1.6, 10000);
        };
        ws.onerror = function() { ws.close(); };
      }

      // Used right after a successful PIN login — the browser's new cookie only takes effect on
      // the *next* connection, not the one already open, so the old one is torn down explicitly
      // (nulling onclose first so its own auto-reconnect doesn't also fire and open a duplicate).
      function reconnectNow() {
        if (ws) { ws.onclose = null; try { ws.close(); } catch (err) {} }
        reconnectDelay = 1000;
        connect();
      }

      function send(obj) {
        if (ws && ws.readyState === 1) { ws.send(JSON.stringify(obj)); }
      }

      function showLogin() {
        loginEl.style.display = 'block';
        chatEl.classList.remove('visible');
      }
      function showChat() {
        loginEl.style.display = 'none';
        chatEl.classList.add('visible');
        statusEl.textContent = 'connected';
      }

      function handle(msg) {
        switch (msg.type) {
          case 'sessionState':
            if (msg.authenticated) {
              loginError.textContent = '';
              showChat();
              send({type: 'listConversations'});
            } else {
              showLogin();
            }
            break;
          case 'conversations':
            conversations = msg.items;
            renderConvList();
            if (!activeId && conversations.length > 0) {
              selectConversation(conversations[0].id);
            }
            break;
          case 'messages':
            if (msg.conversationId === activeId) { renderMessages(msg.items); }
            break;
          case 'created':
            activeId = msg.conversationId;
            send({type: 'listConversations'});
            send({type: 'getMessages', conversationId: activeId});
            break;
          case 'token':
            appendStreamToken(msg.text);
            break;
          case 'done':
            finishStreaming();
            sendBtn.disabled = false;
            break;
          case 'notice':
            appendNoticeBubble(msg.text);
            finishStreaming();
            sendBtn.disabled = false;
            break;
          case 'error':
            appendNoticeBubble(msg.message || 'Something went wrong.');
            sendBtn.disabled = false;
            break;
          case 'settings':
            renderSettings(msg);
            break;
          case 'settingsNotice':
            showSettingsNotice(msg.message, msg.isError);
            break;
          case 'modelSafetyConfirm':
            if (confirm(msg.message + '\\n\\nLoad anyway?')) {
              send({type: 'selectModel', path: msg.path, kind: msg.kind, force: true});
            }
            break;
          case 'modelLoadState':
            // Pushed live on every progress tick (see `WebPortalManager.broadcastLoadState`) —
            // a full `settings` snapshot only arrives once the load actually settles, so this is
            // what keeps the banner and the disabled Load/Unload buttons in sync while it's
            // still in progress.
            updateLoadBanner(!!msg.isLoading, msg.progress, msg.message);
            renderModels(lastModels);
            break;
        }
      }

      function renderConvList() {
        convListEl.innerHTML = '';
        conversations.forEach(function(c) {
          var row = document.createElement('div');
          row.className = 'convItem' + (c.id === activeId ? ' active' : '');
          var title = document.createElement('div');
          title.className = 'convTitle';
          title.textContent = c.title;
          row.appendChild(title);
          var del = document.createElement('button');
          del.className = 'convDelete';
          del.textContent = '✕';
          del.onclick = function(ev) {
            ev.stopPropagation();
            if (confirm('Delete "' + c.title + '"?')) {
              send({type: 'deleteConversation', conversationId: c.id});
              if (c.id === activeId) { activeId = null; }
            }
          };
          row.appendChild(del);
          row.onclick = function() { selectConversation(c.id); closeSidebarOnMobile(); };
          convListEl.appendChild(row);
        });
      }

      function selectConversation(id) {
        activeId = id;
        renderConvList();
        send({type: 'getMessages', conversationId: id});
      }

      function renderMessages(items) {
        messagesEl.innerHTML = '';
        if (items.length === 0) {
          var empty = document.createElement('div');
          empty.id = 'empty';
          empty.textContent = 'Say something to get started.';
          messagesEl.appendChild(empty);
          return;
        }
        items.forEach(function(m) { messagesEl.appendChild(makeBubble(m)); });
        scrollToBottom();
      }

      function makeBubble(m) {
        var row = document.createElement('div');
        row.className = 'msgRow ' + (m.isUser ? 'user' : 'assistant');
        var bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = m.text;
        if (m.imageURL) {
          var img = document.createElement('img');
          img.src = m.imageURL;
          bubble.appendChild(img);
        }
        row.appendChild(bubble);
        return row;
      }

      function appendStreamToken(text) {
        if (!streamingBubble) {
          document.getElementById('empty') && (document.getElementById('empty').remove());
          var row = document.createElement('div');
          row.className = 'msgRow assistant';
          var bubble = document.createElement('div');
          bubble.className = 'bubble streaming';
          row.appendChild(bubble);
          messagesEl.appendChild(row);
          streamingBubble = bubble;
        }
        streamingBubble.textContent += text;
        scrollToBottom();
      }

      function appendNoticeBubble(text) {
        if (!streamingBubble) {
          var row = document.createElement('div');
          row.className = 'msgRow assistant';
          var bubble = document.createElement('div');
          bubble.className = 'bubble';
          row.appendChild(bubble);
          messagesEl.appendChild(row);
          streamingBubble = bubble;
        }
        streamingBubble.textContent = text;
        streamingBubble.classList.remove('streaming');
        streamingBubble = null;
        scrollToBottom();
      }

      function finishStreaming() {
        if (streamingBubble) { streamingBubble.classList.remove('streaming'); }
        streamingBubble = null;
      }

      function scrollToBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }

      function closeSidebarOnMobile() {
        sidebar.classList.remove('open');
        sidebarScrim.classList.remove('visible');
      }

      pinSubmit.onclick = function() {
        var pin = pinInput.value.trim();
        if (!pin) { return; }
        pinSubmit.disabled = true;
        fetch('/api/login', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({pin: pin})
        }).then(function(r) { return r.json(); }).then(function(data) {
          pinSubmit.disabled = false;
          if (data.success) {
            loginError.textContent = '';
            pinInput.value = '';
            reconnectNow();
          } else {
            loginError.textContent = data.message || 'Incorrect PIN.';
          }
        }).catch(function() {
          pinSubmit.disabled = false;
          loginError.textContent = 'Connection error — try again.';
        });
      };
      pinInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { pinSubmit.click(); }
      });

      newChatBtn.onclick = function() { send({type: 'newConversation'}); closeSidebarOnMobile(); };

      menuBtn.onclick = function() {
        sidebar.classList.add('open');
        sidebarScrim.classList.add('visible');
      };
      sidebarScrim.onclick = closeSidebarOnMobile;

      function doSend() {
        var text = textInput.value.trim();
        if (!text || !activeId) { return; }
        textInput.value = '';
        textInput.style.height = 'auto';
        sendBtn.disabled = true;
        var empty = document.getElementById('empty');
        if (empty) { empty.remove(); }
        var row = document.createElement('div');
        row.className = 'msgRow user';
        var bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = text;
        row.appendChild(bubble);
        messagesEl.appendChild(row);
        scrollToBottom();
        send({type: 'send', conversationId: activeId, text: text});
      }
      sendBtn.onclick = doSend;
      textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSend(); }
      });
      textInput.addEventListener('input', function() {
        textInput.style.height = 'auto';
        textInput.style.height = Math.min(textInput.scrollHeight, 140) + 'px';
      });

      function escapeHTML(s) {
        var d = document.createElement('div');
        d.textContent = s == null ? '' : s;
        return d.innerHTML;
      }

      function showSettingsNotice(text, isError) {
        settingsNotice.textContent = text;
        settingsNotice.className = 'visible ' + (isError ? 'error' : 'info');
        clearTimeout(showSettingsNotice._t);
        showSettingsNotice._t = setTimeout(function() { settingsNotice.classList.remove('visible'); }, 4000);
      }

      function makeRangeRow(label, value, min, max, step, format, onChange, disabled) {
        var row = document.createElement('div');
        row.className = 'rangeRow';
        var top = document.createElement('div');
        top.className = 'rangeTop';
        var l = document.createElement('span'); l.textContent = label;
        var v = document.createElement('span'); v.className = 'rangeValue'; v.textContent = format(value);
        top.appendChild(l); top.appendChild(v);
        var input = document.createElement('input');
        input.type = 'range'; input.min = min; input.max = max; input.step = step; input.value = value;
        if (disabled) { input.disabled = true; input.style.opacity = '0.5'; }
        input.oninput = function() { v.textContent = format(parseFloat(input.value)); };
        input.onchange = function() { onChange(parseFloat(input.value)); };
        row.appendChild(top);
        row.appendChild(input);
        return row;
      }

      function renderSettings(s) {
        var ci = document.getElementById('setCustomInstructions');
        if (document.activeElement !== ci) { ci.value = s.customInstructions || ''; }

        document.getElementById('setEnableRAG').checked = !!s.enableRAG;
        document.getElementById('setEnableMemories').checked = !!s.enableMemories;
        document.getElementById('setInternetAccess').checked = !!s.internetAccess;
        document.getElementById('memoriesBody').style.display = s.enableMemories ? 'block' : 'none';
        document.getElementById('internetBody').style.display = s.internetAccess ? 'block' : 'none';
        document.getElementById('braveKeyStatus').textContent = s.hasBraveKey
          ? 'Key saved on this device — full web search is active.'
          : 'No key set — searches use the free weather and Wikipedia lookups.';

        var badge = document.getElementById('personalityBadge');
        if (s.personality) {
          badge.textContent = (s.personality.isMature ? 'ADAPTED' : 'LEARNING…') + ' · ' + s.personality.databaseSize;
        }

        // Ahead of `renderModels` below — it reads `isModelLoading` to decide whether Load/Unload
        // buttons render disabled. Covers a Settings panel opened *while* a load already started
        // elsewhere (the phone, or another browser), not just loads triggered from this panel.
        updateLoadBanner(!!s.isLoading, s.loadingProgress, s.loadingMessage);

        renderLLMSettings(s);
        renderModels(s.models || {llm: [], coreml: []});
        renderMemories(s.memories || []);
        renderFeedback(s.feedback || {upCount: 0, downCount: 0, avoidDirectives: '', downVotes: []});
      }

      function updateLoadBanner(isLoading, progress, message) {
        isModelLoading = isLoading;
        var banner = document.getElementById('modelLoadBanner');
        if (isLoading) {
          var pct = Math.round((progress || 0) * 100);
          banner.innerHTML = '<div>' + escapeHTML(message || 'Loading…') + ' (' + pct + '%)</div>' +
            '<div class="loadBarTrack"><div class="loadBarFill" style="width:' + pct + '%"></div></div>';
          banner.classList.add('visible');
        } else {
          banner.classList.remove('visible');
          banner.innerHTML = '';
        }
      }

      function renderLLMSettings(s) {
        var el = document.getElementById('llmSettingsBody');
        el.innerHTML = '';

        if (s.activeModelName) {
          var active = document.createElement('div');
          active.className = 'settingsSubtle';
          active.style.marginBottom = '10px';
          active.textContent = 'Active: ' + s.activeModelName;
          el.appendChild(active);
        }

        if (s.backend === 'coreML') {
          var p = document.createElement('div');
          p.className = 'settingsSubtle';
          p.textContent = 'Context window: ' + s.coreMLContextWindow + ' tokens' +
            (s.coreMLContextSliding ? ' (sliding — earliest turns are forgotten past this)' : ' (fixed, prompt and reply combined)') +
            '. Not adjustable for Core ML models.';
          el.appendChild(p);
          return;
        }

        el.appendChild(makeRangeRow('Context Window Limit', s.contextLimit || 8192, 512, s.contextCeiling || 32768, 256,
          function(v) { return Math.round(v) + ' tokens'; },
          function(v) { send({type: 'setContextLimit', value: Math.round(v)}); }));

        el.appendChild(makeRangeRow('Max Output Limit', s.maxTokens || 512, 64, 8192, 128,
          function(v) { return Math.round(v) + ' tokens'; },
          function(v) { send({type: 'setMaxTokens', value: Math.round(v)}); }));

        el.appendChild(makeRangeRow('Temperature (Creativity)', s.temperature || 0.85, 0, 2, 0.05,
          function(v) { return v.toFixed(2); },
          function(v) { send({type: 'setTemperature', value: v}); },
          s.highVariability));

        var row = document.createElement('label');
        row.className = 'toggleRow';
        var span = document.createElement('span');
        span.textContent = 'High Variability';
        var sw = document.createElement('span');
        sw.className = 'switch';
        var input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = !!s.highVariability;
        input.onchange = function() { send({type: 'setHighVariability', value: input.checked}); };
        var slider = document.createElement('span');
        slider.className = 'slider';
        sw.appendChild(input);
        sw.appendChild(slider);
        row.appendChild(span);
        row.appendChild(sw);
        el.appendChild(row);
      }

      function renderModels(models) {
        lastModels = models || {llm: [], coreml: []};
        var llmEl = document.getElementById('llmModelList');
        var coremlEl = document.getElementById('coremlModelList');
        llmEl.innerHTML = '';
        coremlEl.innerHTML = '';

        (models.llm || []).forEach(function(m) { llmEl.appendChild(makeModelRow(m, 'llm')); });
        (models.coreml || []).forEach(function(m) { coremlEl.appendChild(makeModelRow(m, 'coreml')); });

        if ((models.llm || []).length === 0 && (models.coreml || []).length === 0) {
          var p = document.createElement('div');
          p.className = 'settingsSubtle';
          p.textContent = 'No models installed yet.';
          llmEl.appendChild(p);
        }
      }

      function makeModelRow(m, kind) {
        var row = document.createElement('div');
        row.className = 'modelRow';
        var info = document.createElement('div');
        var name = document.createElement('div');
        name.style.fontSize = '13px';
        name.style.fontWeight = '600';
        name.textContent = m.name;
        var meta = document.createElement('div');
        meta.className = 'settingsSubtle';
        var metaText = m.sizeGB.toFixed(2) + ' GB' + (kind === 'coreml' ? ' · ANE' : '');
        if (m.safety && m.safety !== 'safe') { metaText += ' · ' + m.safety.toUpperCase(); }
        meta.textContent = metaText;
        info.appendChild(name);
        info.appendChild(meta);
        row.appendChild(info);

        var btn = document.createElement('button');
        btn.className = 'smallBtn';
        btn.textContent = m.isLoaded ? 'Unload' : 'Load';
        if (isModelLoading) {
          btn.disabled = true;
        } else if (m.isLoaded) {
          btn.onclick = function() { send({type: 'unloadModel'}); };
        } else {
          btn.onclick = function() { send({type: 'selectModel', path: m.path, kind: kind}); };
        }
        row.appendChild(btn);
        return row;
      }

      function renderMemories(items) {
        var el = document.getElementById('memoriesBody');
        el.innerHTML = '';

        var header = document.createElement('div');
        header.className = 'settingsRowHeader';
        var label = document.createElement('span');
        label.className = 'settingsSubtle';
        label.textContent = 'EXTRACTED PREFERENCES';
        var clearBtn = document.createElement('button');
        clearBtn.className = 'linkBtn';
        clearBtn.textContent = 'Clear All';
        clearBtn.onclick = function() { if (confirm('Clear all memories?')) { send({type: 'clearMemories'}); } };
        header.appendChild(label);
        header.appendChild(clearBtn);
        el.appendChild(header);

        if (items.length === 0) {
          var p = document.createElement('div');
          p.className = 'settingsSubtle';
          p.style.padding = '4px 0';
          p.textContent = 'No memories extracted yet.';
          el.appendChild(p);
          return;
        }

        items.forEach(function(mem) {
          var row = document.createElement('div');
          row.className = 'settingsRow';
          var left = document.createElement('div');
          left.style.display = 'flex';
          left.style.gap = '8px';
          left.style.alignItems = 'flex-start';
          var badge = document.createElement('span');
          badge.className = 'badge small';
          badge.textContent = mem.kindLabel;
          var text = document.createElement('span');
          text.style.fontSize = '13px';
          text.textContent = mem.text;
          left.appendChild(badge);
          left.appendChild(text);
          var del = document.createElement('button');
          del.className = 'convDelete';
          del.textContent = '✕';
          del.onclick = function() { send({type: 'deleteMemory', id: mem.id}); };
          row.appendChild(left);
          row.appendChild(del);
          el.appendChild(row);
        });
      }

      function renderFeedback(fb) {
        document.getElementById('feedbackCounts').textContent = fb.upCount + ' up · ' + fb.downCount + ' down';
        var el = document.getElementById('feedbackList');
        el.innerHTML = '';

        if (fb.avoidDirectives) {
          var ad = document.createElement('div');
          ad.className = 'settingsSubtle';
          ad.style.marginBottom = '8px';
          ad.textContent = 'Currently avoiding: ' + fb.avoidDirectives;
          el.appendChild(ad);
        }

        if (!(fb.downVotes || []).length) { return; }

        var header = document.createElement('div');
        header.className = 'settingsRowHeader';
        var label = document.createElement('span');
        label.className = 'settingsSubtle';
        label.textContent = 'DOWN-VOTED REPLIES';
        var clearBtn = document.createElement('button');
        clearBtn.className = 'linkBtn';
        clearBtn.textContent = 'Clear All';
        clearBtn.onclick = function() { if (confirm('Clear all feedback?')) { send({type: 'clearFeedback'}); } };
        header.appendChild(label);
        header.appendChild(clearBtn);
        el.appendChild(header);

        fb.downVotes.forEach(function(entry) {
          var row = document.createElement('div');
          row.className = 'settingsRow';
          var text = document.createElement('div');
          text.style.fontSize = '12px';
          text.innerHTML = '<div style="font-weight:600;">' + escapeHTML(entry.userPrompt) + '</div>' +
            '<div class="settingsSubtle">' + escapeHTML(entry.assistantResponse) + '</div>';
          var del = document.createElement('button');
          del.className = 'convDelete';
          del.textContent = '✕';
          del.onclick = function() { send({type: 'deleteFeedback', id: entry.id}); };
          row.appendChild(text);
          row.appendChild(del);
          el.appendChild(row);
        });
      }

      settingsBtn.onclick = function() {
        settingsPanel.classList.add('visible');
        send({type: 'getSettings'});
      };
      settingsClose.onclick = function() { settingsPanel.classList.remove('visible'); };

      document.getElementById('setCustomInstructions').addEventListener('change', function(e) {
        send({type: 'setCustomInstructions', text: e.target.value});
      });
      document.getElementById('setEnableRAG').addEventListener('change', function(e) {
        send({type: 'setEnableRAG', value: e.target.checked});
      });
      document.getElementById('setEnableMemories').addEventListener('change', function(e) {
        document.getElementById('memoriesBody').style.display = e.target.checked ? 'block' : 'none';
        send({type: 'setEnableMemories', value: e.target.checked});
      });
      document.getElementById('setInternetAccess').addEventListener('change', function(e) {
        document.getElementById('internetBody').style.display = e.target.checked ? 'block' : 'none';
        send({type: 'setInternetAccess', value: e.target.checked});
      });
      document.getElementById('setBraveKey').addEventListener('change', function(e) {
        if (e.target.value) { send({type: 'setBraveAPIKey', value: e.target.value}); e.target.value = ''; }
      });
      document.getElementById('resetPersonalityBtn').onclick = function() {
        if (confirm('Reset personality? This erases every learned speech pattern, for all models.')) {
          send({type: 'resetPersonality'});
        }
      };

      showLogin();
      connect();
    })();
    </script>
    </body>
    </html>
    """
}
