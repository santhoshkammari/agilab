# Ubuntu 24.04 — Catppuccin Mocha Theme Setup

A complete, reproducible guide to achieve a unified Catppuccin Mocha aesthetic across your entire Ubuntu 24.04 desktop — GTK theme, icons, terminal, VS Code, Firefox, wallpaper, and fonts. Works on any machine running Ubuntu 24.04 with GNOME.

---

## What You'll Get

| Layer | Result |
|---|---|
| GTK / Window theme | Catppuccin Mocha Mauve |
| Shell | Dark mode |
| Icons | Papirus-Dark with Catppuccin Mocha Mauve folders |
| Terminal | Catppuccin Mocha color palette |
| VS Code | Catppuccin Mocha theme + icons + JetBrains Mono |
| Firefox | Catppuccin Mocha via userChrome.css |
| Wallpaper | Catppuccin Mocha wallpaper |
| Fonts | Ubuntu 11 (UI) · Ubuntu Mono 12 (terminal) |

---

## Prerequisites

```bash
sudo apt update
sudo apt install -y gnome-tweaks unzip wget curl python3
```

Make sure the **User Themes** GNOME extension is enabled:

```bash
gnome-extensions enable user-theme@gnome-shell-extensions.gcampax.github.com
```

---

## Step 1 — Fix Display Scaling (Important First)

Ubuntu sometimes ships with broken font scaling. Reset it to the correct baseline:

```bash
gsettings set org.gnome.desktop.interface text-scaling-factor 1.0
gsettings set org.gnome.desktop.interface font-name 'Ubuntu 11'
gsettings set org.gnome.desktop.interface document-font-name 'Ubuntu 11'
gsettings set org.gnome.desktop.interface monospace-font-name 'Ubuntu Mono 12'
```

> **Why:** A scaling factor below 1.0 (e.g. 0.7) shrinks text to 70% of normal — causing eye strain and blurry-looking text. Always start from 1.0.

---

## Step 2 — Install Catppuccin GTK Theme

```bash
mkdir -p ~/.themes
cd /tmp
wget "https://github.com/catppuccin/gtk/releases/download/v1.0.3/catppuccin-mocha-mauve-standard%2Bdefault.zip" \
  -O catppuccin-mocha-mauve.zip
unzip -q catppuccin-mocha-mauve.zip -d ~/.themes/
```

Verify:

```bash
ls ~/.themes | grep catppuccin
# Expected: catppuccin-mocha-mauve-standard+default
```

---

## Step 3 — Install Latest Papirus Icons + Catppuccin Folders

The apt version of Papirus is outdated and lacks Catppuccin folder colors. Install from source:

```bash
# Install latest Papirus icon theme
cd /tmp
curl -L "https://api.github.com/repos/PapirusDevelopmentTeam/papirus-icon-theme/tarball/20250501" \
  -o papirus-latest.tar.gz
tar -xzf papirus-latest.tar.gz
cd PapirusDevelopmentTeam-papirus-icon-theme-*
sudo ./install.sh
```

Download the Catppuccin folder overlay and papirus-folders script:

```bash
cd /tmp
wget https://raw.githubusercontent.com/PapirusDevelopmentTeam/papirus-folders/master/papirus-folders \
  -O papirus-folders && chmod +x papirus-folders

wget "https://github.com/catppuccin/papirus-folders/archive/refs/heads/main.tar.gz" \
  -O catppuccin-papirus.tar.gz
tar -xzf catppuccin-papirus.tar.gz
```

Copy Catppuccin Mocha Mauve folder icons into Papirus-Dark:

```bash
for size in 22x22 24x24 32x32 48x48 64x64; do
  sudo cp /tmp/papirus-folders-main/src/$size/places/folder-cat-mocha-mauve*.svg \
    /usr/share/icons/Papirus-Dark/$size/places/ 2>/dev/null
done
```

Apply the folder color:

```bash
sudo /tmp/papirus-folders -C cat-mocha-mauve --theme Papirus-Dark
```

Refresh icon cache:

```bash
sudo gtk-update-icon-cache /usr/share/icons/Papirus-Dark/ -f
```

---

## Step 4 — Apply GNOME Settings

```bash
# GTK theme
gsettings set org.gnome.desktop.interface gtk-theme 'catppuccin-mocha-mauve-standard+default'

# Shell theme (requires User Themes extension)
gsettings set org.gnome.shell.extensions.user-theme name 'catppuccin-mocha-mauve-standard+default'

# Dark mode
gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'

# Icons
gsettings set org.gnome.desktop.interface icon-theme 'Papirus-Dark'

# Fonts
gsettings set org.gnome.desktop.interface font-name 'Ubuntu 11'
gsettings set org.gnome.desktop.interface document-font-name 'Ubuntu 11'
gsettings set org.gnome.desktop.interface monospace-font-name 'Ubuntu Mono 12'
gsettings set org.gnome.desktop.interface text-scaling-factor 1.0
```

---

## Step 5 — Terminal Colors (GNOME Terminal)

```bash
PROFILE=$(gsettings get org.gnome.Terminal.ProfilesList default | tr -d "'")
BASE="org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:$PROFILE/"

gsettings set $BASE use-theme-colors false
gsettings set $BASE use-transparent-background false
gsettings set $BASE background-color '#1E1E2E'
gsettings set $BASE foreground-color '#CDD6F4'
gsettings set $BASE bold-color '#CDD6F4'
gsettings set $BASE bold-color-same-as-fg true
gsettings set $BASE cursor-background-color '#F5E0DC'
gsettings set $BASE cursor-foreground-color '#1E1E2E'
gsettings set $BASE highlight-background-color '#313244'
gsettings set $BASE highlight-foreground-color '#CDD6F4'
gsettings set $BASE palette "['#45475A', '#F38BA8', '#A6E3A1', '#F9E2AF', '#89B4FA', '#CBA6F7', '#89DCEB', '#BAC2DE', '#585B70', '#F38BA8', '#A6E3A1', '#F9E2AF', '#89B4FA', '#CBA6F7', '#89DCEB', '#A6ADC8']"
gsettings set $BASE font 'Ubuntu Mono 12'
gsettings set $BASE use-system-font false
```

Reopen your terminal to see the new colors.

---

## Step 6 — VS Code Theme

```bash
code --install-extension Catppuccin.catppuccin-vsc
code --install-extension Catppuccin.catppuccin-vsc-icons
```

Apply settings (adds to your existing `settings.json`):

```bash
python3 - <<'EOF'
import json, os
path = os.path.expanduser('~/.config/Code/User/settings.json')
try:
    with open(path) as f:
        s = json.load(f)
except:
    s = {}
s['workbench.colorTheme'] = 'Catppuccin Mocha'
s['workbench.iconTheme'] = 'catppuccin-mocha'
s['editor.fontFamily'] = "'JetBrains Mono', 'Ubuntu Mono', monospace"
s['editor.fontSize'] = 13
s['editor.fontLigatures'] = True
with open(path, 'w') as f:
    json.dump(s, f, indent=2)
print('VS Code settings updated')
EOF
```

Reload VS Code (`Ctrl+Shift+P` → `Developer: Reload Window`).

---

## Step 7 — Wallpaper

```bash
mkdir -p ~/Pictures/Wallpapers
wget "https://raw.githubusercontent.com/zhichaoh/catppuccin-wallpapers/main/os/catppuccin-mocha.png" \
  -O ~/Pictures/Wallpapers/catppuccin-mocha.png

gsettings set org.gnome.desktop.background picture-uri \
  "file://$HOME/Pictures/Wallpapers/catppuccin-mocha.png"
gsettings set org.gnome.desktop.background picture-uri-dark \
  "file://$HOME/Pictures/Wallpapers/catppuccin-mocha.png"
gsettings set org.gnome.desktop.background picture-options 'zoom'
```

---

## Step 8 — Firefox Theme

### Option A — Add-on (Easiest)
Install the official Catppuccin Mocha Mauve theme directly from Firefox Add-ons:
1. Open Firefox and go to `https://addons.mozilla.org/firefox/addon/catppuccin-mocha-mauve/`
2. Click **Add to Firefox**

### Option B — userChrome.css (Deeper customization)

Enable custom CSS in Firefox first:
1. Open Firefox, go to `about:config`
2. Search for `toolkit.legacyUserProfileCustomizations.stylesheets` → set to `true`

Find your Firefox profile folder:
```bash
firefox_profile=$(find ~/.mozilla/firefox -maxdepth 1 -name "*.default-release" -o -name "*.default" | head -1)
echo "Profile: $firefox_profile"
mkdir -p "$firefox_profile/chrome"
```

Create `userChrome.css`:
```bash
cat > "$firefox_profile/chrome/userChrome.css" << 'EOF'
/* Catppuccin Mocha — Firefox userChrome */
:root {
  --ctp-base:    #1E1E2E;
  --ctp-mantle:  #181825;
  --ctp-crust:   #11111B;
  --ctp-text:    #CDD6F4;
  --ctp-mauve:   #CBA6F7;
  --ctp-surface0:#313244;
  --ctp-surface1:#45475A;
  --ctp-overlay0:#6C7086;
}

/* Tab bar & toolbar background */
#navigator-toolbox {
  background-color: var(--ctp-mantle) !important;
  border-bottom: 1px solid var(--ctp-surface0) !important;
}

/* Active tab */
.tab-background[selected] {
  background-color: var(--ctp-base) !important;
}

/* Inactive tabs */
.tab-background {
  background-color: var(--ctp-crust) !important;
}

/* Tab text */
.tab-label {
  color: var(--ctp-text) !important;
}

/* URL bar */
#urlbar-background {
  background-color: var(--ctp-surface0) !important;
  border-color: var(--ctp-surface1) !important;
}
#urlbar-input {
  color: var(--ctp-text) !important;
}

/* Sidebar */
#sidebar-box {
  background-color: var(--ctp-mantle) !important;
}
EOF
```

Restart Firefox to apply.

---

## Step 9 — Blur My Shell (Optional but Recommended)

Adds frosted glass blur to the GNOME top panel and overview.

Install via browser:
1. Open Firefox and go to: `https://extensions.gnome.org/extension/3193/blur-my-shell/`
2. Toggle the extension **ON**
3. Configure: open **Extension Manager** → Blur My Shell → set blur intensity to ~20

> Requires the GNOME Shell browser integration. If prompted, install the browser extension and native connector (`sudo apt install chrome-gnome-shell`).

---

## All-in-One Script

Save as `setup-catppuccin.sh` and run with `bash setup-catppuccin.sh`:

```bash
#!/bin/bash
set -e

echo "==> Step 1: Fix font scaling"
gsettings set org.gnome.desktop.interface text-scaling-factor 1.0
gsettings set org.gnome.desktop.interface font-name 'Ubuntu 11'
gsettings set org.gnome.desktop.interface document-font-name 'Ubuntu 11'
gsettings set org.gnome.desktop.interface monospace-font-name 'Ubuntu Mono 12'

echo "==> Step 2: Install GTK theme"
mkdir -p ~/.themes
cd /tmp
wget -q "https://github.com/catppuccin/gtk/releases/download/v1.0.3/catppuccin-mocha-mauve-standard%2Bdefault.zip" \
  -O catppuccin-mocha-mauve.zip
unzip -q catppuccin-mocha-mauve.zip -d ~/.themes/

echo "==> Step 3: Install latest Papirus icons"
wget -q "https://api.github.com/repos/PapirusDevelopmentTeam/papirus-icon-theme/tarball/20250501" \
  -O papirus-latest.tar.gz
tar -xzf papirus-latest.tar.gz
cd PapirusDevelopmentTeam-papirus-icon-theme-*
sudo ./install.sh
cd /tmp

echo "==> Step 3b: Apply Catppuccin Mocha Mauve folders"
wget -q https://raw.githubusercontent.com/PapirusDevelopmentTeam/papirus-folders/master/papirus-folders \
  -O papirus-folders && chmod +x papirus-folders
wget -q "https://github.com/catppuccin/papirus-folders/archive/refs/heads/main.tar.gz" \
  -O catppuccin-papirus.tar.gz
tar -xzf catppuccin-papirus.tar.gz
for size in 22x22 24x24 32x32 48x48 64x64; do
  sudo cp /tmp/papirus-folders-main/src/$size/places/folder-cat-mocha-mauve*.svg \
    /usr/share/icons/Papirus-Dark/$size/places/ 2>/dev/null || true
done
sudo /tmp/papirus-folders -C cat-mocha-mauve --theme Papirus-Dark
sudo gtk-update-icon-cache /usr/share/icons/Papirus-Dark/ -f

echo "==> Step 4: Apply GNOME settings"
gnome-extensions enable user-theme@gnome-shell-extensions.gcampax.github.com 2>/dev/null || true
gsettings set org.gnome.desktop.interface gtk-theme 'catppuccin-mocha-mauve-standard+default'
gsettings set org.gnome.shell.extensions.user-theme name 'catppuccin-mocha-mauve-standard+default'
gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'
gsettings set org.gnome.desktop.interface icon-theme 'Papirus-Dark'

echo "==> Step 5: Terminal colors"
PROFILE=$(gsettings get org.gnome.Terminal.ProfilesList default | tr -d "'")
BASE="org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:$PROFILE/"
gsettings set $BASE use-theme-colors false
gsettings set $BASE use-transparent-background false
gsettings set $BASE background-color '#1E1E2E'
gsettings set $BASE foreground-color '#CDD6F4'
gsettings set $BASE bold-color '#CDD6F4'
gsettings set $BASE bold-color-same-as-fg true
gsettings set $BASE cursor-background-color '#F5E0DC'
gsettings set $BASE cursor-foreground-color '#1E1E2E'
gsettings set $BASE highlight-background-color '#313244'
gsettings set $BASE highlight-foreground-color '#CDD6F4'
gsettings set $BASE palette "['#45475A', '#F38BA8', '#A6E3A1', '#F9E2AF', '#89B4FA', '#CBA6F7', '#89DCEB', '#BAC2DE', '#585B70', '#F38BA8', '#A6E3A1', '#F9E2AF', '#89B4FA', '#CBA6F7', '#89DCEB', '#A6ADC8']"
gsettings set $BASE font 'Ubuntu Mono 12'
gsettings set $BASE use-system-font false

echo "==> Step 6: VS Code"
if command -v code &>/dev/null; then
  code --install-extension Catppuccin.catppuccin-vsc
  code --install-extension Catppuccin.catppuccin-vsc-icons
  python3 - <<'PYEOF'
import json, os
path = os.path.expanduser('~/.config/Code/User/settings.json')
try:
    with open(path) as f:
        s = json.load(f)
except:
    s = {}
s['workbench.colorTheme'] = 'Catppuccin Mocha'
s['workbench.iconTheme'] = 'catppuccin-mocha'
s["editor.fontFamily"] = "'JetBrains Mono', 'Ubuntu Mono', monospace"
s['editor.fontSize'] = 13
s['editor.fontLigatures'] = True
with open(path, 'w') as f:
    json.dump(s, f, indent=2)
print('VS Code settings updated')
PYEOF
fi

echo "==> Step 7: Firefox theme"
firefox_profile=$(find ~/.mozilla/firefox -maxdepth 1 -name "*.default-release" -o -name "*.default" 2>/dev/null | head -1)
if [ -n "$firefox_profile" ]; then
  mkdir -p "$firefox_profile/chrome"
  cat > "$firefox_profile/chrome/userChrome.css" << 'FFEOF'
:root {
  --ctp-base:    #1E1E2E;
  --ctp-mantle:  #181825;
  --ctp-crust:   #11111B;
  --ctp-text:    #CDD6F4;
  --ctp-mauve:   #CBA6F7;
  --ctp-surface0:#313244;
  --ctp-surface1:#45475A;
}
#navigator-toolbox { background-color: var(--ctp-mantle) !important; border-bottom: 1px solid var(--ctp-surface0) !important; }
.tab-background[selected] { background-color: var(--ctp-base) !important; }
.tab-background { background-color: var(--ctp-crust) !important; }
.tab-label { color: var(--ctp-text) !important; }
#urlbar-background { background-color: var(--ctp-surface0) !important; border-color: var(--ctp-surface1) !important; }
#urlbar-input { color: var(--ctp-text) !important; }
#sidebar-box { background-color: var(--ctp-mantle) !important; }
FFEOF
  # Enable userChrome in Firefox prefs
  echo 'user_pref("toolkit.legacyUserProfileCustomizations.stylesheets", true);' >> "$firefox_profile/user.js"
  echo "Firefox userChrome.css applied to $firefox_profile"
else
  echo "Firefox profile not found — open Firefox once then re-run this step"
fi

echo "==> Step 8: Wallpaper"
mkdir -p ~/Pictures/Wallpapers
wget -q "https://raw.githubusercontent.com/zhichaoh/catppuccin-wallpapers/main/os/catppuccin-mocha.png" \
  -O ~/Pictures/Wallpapers/catppuccin-mocha.png
gsettings set org.gnome.desktop.background picture-uri \
  "file://$HOME/Pictures/Wallpapers/catppuccin-mocha.png"
gsettings set org.gnome.desktop.background picture-uri-dark \
  "file://$HOME/Pictures/Wallpapers/catppuccin-mocha.png"
gsettings set org.gnome.desktop.background picture-options 'zoom'

echo ""
echo "✓ Catppuccin Mocha setup complete!"
echo "  → Reopen terminal to see new colors"
echo "  → Reload VS Code (Ctrl+Shift+P → Developer: Reload Window)"
echo "  → Restart Firefox to apply userChrome.css"
echo "  → Install 'Blur My Shell' from extensions.gnome.org for frosted glass panel"
```

---

## Color Reference — Catppuccin Mocha Palette

| Name | Hex | Use |
|---|---|---|
| Base | `#1E1E2E` | Background |
| Text | `#CDD6F4` | Foreground |
| Mauve | `#CBA6F7` | Accent / folders |
| Blue | `#89B4FA` | Links |
| Green | `#A6E3A1` | Success |
| Red | `#F38BA8` | Errors |
| Yellow | `#F9E2AF` | Warnings |
| Rosewater | `#F5E0DC` | Cursor |

---

*Setup verified on Ubuntu 24.04 LTS · GNOME 46*
