Create this file:

```bash
~/.local/share/applications/spotlight-llm.desktop
```

Paste **exactly** this (adjust paths if needed):

```ini
[Desktop Entry]
Name=Spotlight LLM
Comment=Run Spotlight LLM
Exec=/home/ntlpt24/.venv/bin/python /home/ntlpt24/utils/spotlight/app.py
Type=Application
Terminal=false
Categories=Utility;
```

---

### 2️⃣ Make it executable

This step is mandatory.

```bash
chmod +x ~/.local/share/applications/spotlight-llm.desktop
```

---

### 3️⃣ Trust the launcher (important!)

Ubuntu hides untrusted launchers.

Run:

```bash
gio set ~/.local/share/applications/spotlight-llm.desktop metadata::trusted true
```

Or:

* Right-click the `.desktop` file
* Properties → **Allow Launching**

---

### 4️⃣ Refresh app database

Usually automatic, but just in case:

```bash
update-desktop-database ~/.local/share/applications/
```

---

### 5️⃣ Launch it 🎉

* Press **Super (Windows key)**
* Type **Spotlight LLM**
* Click it — it should run like a normal app

---

## Optional but recommended tweaks 🔧

### Add an icon

Put an icon somewhere, e.g.:

```bash
~/utils/spotlight/icon.png
```

Then add this line:

```ini
Icon=/home/ntlpt24/utils/spotlight/icon.png
```

---

### If it doesn’t start

Run this to debug:

```bash
gtk-launch spotlight-llm
```

Or try:

```bash
Exec=/bin/bash -c "/home/ntlpt24/.venv/bin/python /home/ntlpt24/utils/spotlight/app.py"
```

(useful if env vars are missing)

---

## Result

You’ll have:

* A real Ubuntu app
* Searchable from Activities
* No terminal popping up
* One-click launch

If you want it pinned to the dock, auto-start on login, or packaged as a `.deb` — say the word 👌

