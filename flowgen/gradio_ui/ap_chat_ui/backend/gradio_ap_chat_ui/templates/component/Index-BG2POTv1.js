var Du = Object.defineProperty;
var bi = (n) => {
  throw TypeError(n);
};
var Su = (n, e, t) => e in n ? Du(n, e, { enumerable: !0, configurable: !0, writable: !0, value: t }) : n[e] = t;
var _e = (n, e, t) => Su(n, typeof e != "symbol" ? e + "" : e, t), Au = (n, e, t) => e.has(n) || bi("Cannot " + t);
var yi = (n, e, t) => e.has(n) ? bi("Cannot add the same private member more than once") : e instanceof WeakSet ? e.add(n) : e.set(n, t);
var _r = (n, e, t) => (Au(n, e, "access private method"), t);
const {
  SvelteComponent: Eu,
  append_hydration: Jn,
  assign: Fu,
  attr: Le,
  binding_callbacks: Cu,
  children: J0,
  claim_element: _s,
  claim_space: bs,
  claim_svg_element: xn,
  create_slot: Tu,
  detach: Mt,
  element: ys,
  empty: wi,
  get_all_dirty_from_scope: $u,
  get_slot_changes: Mu,
  get_spread_update: zu,
  init: Bu,
  insert_hydration: lr,
  listen: Ru,
  noop: Nu,
  safe_not_equal: qu,
  set_dynamic_element_data: xi,
  set_style: he,
  space: ws,
  svg_element: kn,
  toggle_class: Be,
  transition_in: xs,
  transition_out: ks,
  update_slot_base: Lu
} = window.__gradio__svelte__internal;
function ki(n) {
  let e, t, r, a, i;
  return {
    c() {
      e = kn("svg"), t = kn("line"), r = kn("line"), this.h();
    },
    l(l) {
      e = xn(l, "svg", { class: !0, xmlns: !0, viewBox: !0 });
      var s = J0(e);
      t = xn(s, "line", {
        x1: !0,
        y1: !0,
        x2: !0,
        y2: !0,
        stroke: !0,
        "stroke-width": !0
      }), J0(t).forEach(Mt), r = xn(s, "line", {
        x1: !0,
        y1: !0,
        x2: !0,
        y2: !0,
        stroke: !0,
        "stroke-width": !0
      }), J0(r).forEach(Mt), s.forEach(Mt), this.h();
    },
    h() {
      Le(t, "x1", "1"), Le(t, "y1", "9"), Le(t, "x2", "9"), Le(t, "y2", "1"), Le(t, "stroke", "gray"), Le(t, "stroke-width", "0.5"), Le(r, "x1", "5"), Le(r, "y1", "9"), Le(r, "x2", "9"), Le(r, "y2", "5"), Le(r, "stroke", "gray"), Le(r, "stroke-width", "0.5"), Le(e, "class", "resize-handle svelte-239wnu"), Le(e, "xmlns", "http://www.w3.org/2000/svg"), Le(e, "viewBox", "0 0 10 10");
    },
    m(l, s) {
      lr(l, e, s), Jn(e, t), Jn(e, r), a || (i = Ru(
        e,
        "mousedown",
        /*resize*/
        n[27]
      ), a = !0);
    },
    p: Nu,
    d(l) {
      l && Mt(e), a = !1, i();
    }
  };
}
function Iu(n) {
  var g;
  let e, t, r, a, i;
  const l = (
    /*#slots*/
    n[31].default
  ), s = Tu(
    l,
    n,
    /*$$scope*/
    n[30],
    null
  );
  let u = (
    /*resizable*/
    n[19] && ki(n)
  ), h = [
    { "data-testid": (
      /*test_id*/
      n[11]
    ) },
    { id: (
      /*elem_id*/
      n[6]
    ) },
    {
      class: r = "block " + /*elem_classes*/
      (((g = n[7]) == null ? void 0 : g.join(" ")) || "") + " svelte-239wnu"
    },
    {
      dir: a = /*rtl*/
      n[20] ? "rtl" : "ltr"
    }
  ], d = {};
  for (let p = 0; p < h.length; p += 1)
    d = Fu(d, h[p]);
  return {
    c() {
      e = ys(
        /*tag*/
        n[25]
      ), s && s.c(), t = ws(), u && u.c(), this.h();
    },
    l(p) {
      e = _s(
        p,
        /*tag*/
        (n[25] || "null").toUpperCase(),
        {
          "data-testid": !0,
          id: !0,
          class: !0,
          dir: !0
        }
      );
      var v = J0(e);
      s && s.l(v), t = bs(v), u && u.l(v), v.forEach(Mt), this.h();
    },
    h() {
      xi(
        /*tag*/
        n[25]
      )(e, d), Be(
        e,
        "hidden",
        /*visible*/
        n[14] === !1
      ), Be(
        e,
        "padded",
        /*padding*/
        n[10]
      ), Be(
        e,
        "flex",
        /*flex*/
        n[1]
      ), Be(
        e,
        "border_focus",
        /*border_mode*/
        n[9] === "focus"
      ), Be(
        e,
        "border_contrast",
        /*border_mode*/
        n[9] === "contrast"
      ), Be(e, "hide-container", !/*explicit_call*/
      n[12] && !/*container*/
      n[13]), Be(
        e,
        "fullscreen",
        /*fullscreen*/
        n[0]
      ), Be(
        e,
        "animating",
        /*fullscreen*/
        n[0] && /*preexpansionBoundingRect*/
        n[24] !== null
      ), Be(
        e,
        "auto-margin",
        /*scale*/
        n[17] === null
      ), he(
        e,
        "height",
        /*fullscreen*/
        n[0] ? void 0 : (
          /*get_dimension*/
          n[26](
            /*height*/
            n[2]
          )
        )
      ), he(
        e,
        "min-height",
        /*fullscreen*/
        n[0] ? void 0 : (
          /*get_dimension*/
          n[26](
            /*min_height*/
            n[3]
          )
        )
      ), he(
        e,
        "max-height",
        /*fullscreen*/
        n[0] ? void 0 : (
          /*get_dimension*/
          n[26](
            /*max_height*/
            n[4]
          )
        )
      ), he(
        e,
        "--start-top",
        /*preexpansionBoundingRect*/
        n[24] ? `${/*preexpansionBoundingRect*/
        n[24].top}px` : "0px"
      ), he(
        e,
        "--start-left",
        /*preexpansionBoundingRect*/
        n[24] ? `${/*preexpansionBoundingRect*/
        n[24].left}px` : "0px"
      ), he(
        e,
        "--start-width",
        /*preexpansionBoundingRect*/
        n[24] ? `${/*preexpansionBoundingRect*/
        n[24].width}px` : "0px"
      ), he(
        e,
        "--start-height",
        /*preexpansionBoundingRect*/
        n[24] ? `${/*preexpansionBoundingRect*/
        n[24].height}px` : "0px"
      ), he(
        e,
        "width",
        /*fullscreen*/
        n[0] ? void 0 : typeof /*width*/
        n[5] == "number" ? `calc(min(${/*width*/
        n[5]}px, 100%))` : (
          /*get_dimension*/
          n[26](
            /*width*/
            n[5]
          )
        )
      ), he(
        e,
        "border-style",
        /*variant*/
        n[8]
      ), he(
        e,
        "overflow",
        /*allow_overflow*/
        n[15] ? (
          /*overflow_behavior*/
          n[16]
        ) : "hidden"
      ), he(
        e,
        "flex-grow",
        /*scale*/
        n[17]
      ), he(e, "min-width", `calc(min(${/*min_width*/
      n[18]}px, 100%))`), he(e, "border-width", "var(--block-border-width)");
    },
    m(p, v) {
      lr(p, e, v), s && s.m(e, null), Jn(e, t), u && u.m(e, null), n[32](e), i = !0;
    },
    p(p, v) {
      var k;
      s && s.p && (!i || v[0] & /*$$scope*/
      1073741824) && Lu(
        s,
        l,
        p,
        /*$$scope*/
        p[30],
        i ? Mu(
          l,
          /*$$scope*/
          p[30],
          v,
          null
        ) : $u(
          /*$$scope*/
          p[30]
        ),
        null
      ), /*resizable*/
      p[19] ? u ? u.p(p, v) : (u = ki(p), u.c(), u.m(e, null)) : u && (u.d(1), u = null), xi(
        /*tag*/
        p[25]
      )(e, d = zu(h, [
        (!i || v[0] & /*test_id*/
        2048) && { "data-testid": (
          /*test_id*/
          p[11]
        ) },
        (!i || v[0] & /*elem_id*/
        64) && { id: (
          /*elem_id*/
          p[6]
        ) },
        (!i || v[0] & /*elem_classes*/
        128 && r !== (r = "block " + /*elem_classes*/
        (((k = p[7]) == null ? void 0 : k.join(" ")) || "") + " svelte-239wnu")) && { class: r },
        (!i || v[0] & /*rtl*/
        1048576 && a !== (a = /*rtl*/
        p[20] ? "rtl" : "ltr")) && { dir: a }
      ])), Be(
        e,
        "hidden",
        /*visible*/
        p[14] === !1
      ), Be(
        e,
        "padded",
        /*padding*/
        p[10]
      ), Be(
        e,
        "flex",
        /*flex*/
        p[1]
      ), Be(
        e,
        "border_focus",
        /*border_mode*/
        p[9] === "focus"
      ), Be(
        e,
        "border_contrast",
        /*border_mode*/
        p[9] === "contrast"
      ), Be(e, "hide-container", !/*explicit_call*/
      p[12] && !/*container*/
      p[13]), Be(
        e,
        "fullscreen",
        /*fullscreen*/
        p[0]
      ), Be(
        e,
        "animating",
        /*fullscreen*/
        p[0] && /*preexpansionBoundingRect*/
        p[24] !== null
      ), Be(
        e,
        "auto-margin",
        /*scale*/
        p[17] === null
      ), v[0] & /*fullscreen, height*/
      5 && he(
        e,
        "height",
        /*fullscreen*/
        p[0] ? void 0 : (
          /*get_dimension*/
          p[26](
            /*height*/
            p[2]
          )
        )
      ), v[0] & /*fullscreen, min_height*/
      9 && he(
        e,
        "min-height",
        /*fullscreen*/
        p[0] ? void 0 : (
          /*get_dimension*/
          p[26](
            /*min_height*/
            p[3]
          )
        )
      ), v[0] & /*fullscreen, max_height*/
      17 && he(
        e,
        "max-height",
        /*fullscreen*/
        p[0] ? void 0 : (
          /*get_dimension*/
          p[26](
            /*max_height*/
            p[4]
          )
        )
      ), v[0] & /*preexpansionBoundingRect*/
      16777216 && he(
        e,
        "--start-top",
        /*preexpansionBoundingRect*/
        p[24] ? `${/*preexpansionBoundingRect*/
        p[24].top}px` : "0px"
      ), v[0] & /*preexpansionBoundingRect*/
      16777216 && he(
        e,
        "--start-left",
        /*preexpansionBoundingRect*/
        p[24] ? `${/*preexpansionBoundingRect*/
        p[24].left}px` : "0px"
      ), v[0] & /*preexpansionBoundingRect*/
      16777216 && he(
        e,
        "--start-width",
        /*preexpansionBoundingRect*/
        p[24] ? `${/*preexpansionBoundingRect*/
        p[24].width}px` : "0px"
      ), v[0] & /*preexpansionBoundingRect*/
      16777216 && he(
        e,
        "--start-height",
        /*preexpansionBoundingRect*/
        p[24] ? `${/*preexpansionBoundingRect*/
        p[24].height}px` : "0px"
      ), v[0] & /*fullscreen, width*/
      33 && he(
        e,
        "width",
        /*fullscreen*/
        p[0] ? void 0 : typeof /*width*/
        p[5] == "number" ? `calc(min(${/*width*/
        p[5]}px, 100%))` : (
          /*get_dimension*/
          p[26](
            /*width*/
            p[5]
          )
        )
      ), v[0] & /*variant*/
      256 && he(
        e,
        "border-style",
        /*variant*/
        p[8]
      ), v[0] & /*allow_overflow, overflow_behavior*/
      98304 && he(
        e,
        "overflow",
        /*allow_overflow*/
        p[15] ? (
          /*overflow_behavior*/
          p[16]
        ) : "hidden"
      ), v[0] & /*scale*/
      131072 && he(
        e,
        "flex-grow",
        /*scale*/
        p[17]
      ), v[0] & /*min_width*/
      262144 && he(e, "min-width", `calc(min(${/*min_width*/
      p[18]}px, 100%))`);
    },
    i(p) {
      i || (xs(s, p), i = !0);
    },
    o(p) {
      ks(s, p), i = !1;
    },
    d(p) {
      p && Mt(e), s && s.d(p), u && u.d(), n[32](null);
    }
  };
}
function Di(n) {
  let e;
  return {
    c() {
      e = ys("div"), this.h();
    },
    l(t) {
      e = _s(t, "DIV", { class: !0 }), J0(e).forEach(Mt), this.h();
    },
    h() {
      Le(e, "class", "placeholder svelte-239wnu"), he(
        e,
        "height",
        /*placeholder_height*/
        n[22] + "px"
      ), he(
        e,
        "width",
        /*placeholder_width*/
        n[23] + "px"
      );
    },
    m(t, r) {
      lr(t, e, r);
    },
    p(t, r) {
      r[0] & /*placeholder_height*/
      4194304 && he(
        e,
        "height",
        /*placeholder_height*/
        t[22] + "px"
      ), r[0] & /*placeholder_width*/
      8388608 && he(
        e,
        "width",
        /*placeholder_width*/
        t[23] + "px"
      );
    },
    d(t) {
      t && Mt(e);
    }
  };
}
function Ou(n) {
  let e, t, r, a = (
    /*tag*/
    n[25] && Iu(n)
  ), i = (
    /*fullscreen*/
    n[0] && Di(n)
  );
  return {
    c() {
      a && a.c(), e = ws(), i && i.c(), t = wi();
    },
    l(l) {
      a && a.l(l), e = bs(l), i && i.l(l), t = wi();
    },
    m(l, s) {
      a && a.m(l, s), lr(l, e, s), i && i.m(l, s), lr(l, t, s), r = !0;
    },
    p(l, s) {
      /*tag*/
      l[25] && a.p(l, s), /*fullscreen*/
      l[0] ? i ? i.p(l, s) : (i = Di(l), i.c(), i.m(t.parentNode, t)) : i && (i.d(1), i = null);
    },
    i(l) {
      r || (xs(a, l), r = !0);
    },
    o(l) {
      ks(a, l), r = !1;
    },
    d(l) {
      l && (Mt(e), Mt(t)), a && a.d(l), i && i.d(l);
    }
  };
}
function Pu(n, e, t) {
  let { $$slots: r = {}, $$scope: a } = e, { height: i = void 0 } = e, { min_height: l = void 0 } = e, { max_height: s = void 0 } = e, { width: u = void 0 } = e, { elem_id: h = "" } = e, { elem_classes: d = [] } = e, { variant: g = "solid" } = e, { border_mode: p = "base" } = e, { padding: v = !0 } = e, { type: k = "normal" } = e, { test_id: A = void 0 } = e, { explicit_call: C = !1 } = e, { container: z = !0 } = e, { visible: x = !0 } = e, { allow_overflow: _ = !0 } = e, { overflow_behavior: w = "auto" } = e, { scale: E = null } = e, { min_width: T = 0 } = e, { flex: $ = !1 } = e, { resizable: M = !1 } = e, { rtl: B = !1 } = e, { fullscreen: G = !1 } = e, U = G, j, oe = k === "fieldset" ? "fieldset" : "div", ee = 0, ue = 0, fe = null;
  function Ee(N) {
    G && N.key === "Escape" && t(0, G = !1);
  }
  const ne = (N) => {
    if (N !== void 0) {
      if (typeof N == "number")
        return N + "px";
      if (typeof N == "string")
        return N;
    }
  }, ve = (N) => {
    let se = N.clientY;
    const ce = (O) => {
      const Ie = O.clientY - se;
      se = O.clientY, t(21, j.style.height = `${j.offsetHeight + Ie}px`, j);
    }, Ce = () => {
      window.removeEventListener("mousemove", ce), window.removeEventListener("mouseup", Ce);
    };
    window.addEventListener("mousemove", ce), window.addEventListener("mouseup", Ce);
  };
  function we(N) {
    Cu[N ? "unshift" : "push"](() => {
      j = N, t(21, j);
    });
  }
  return n.$$set = (N) => {
    "height" in N && t(2, i = N.height), "min_height" in N && t(3, l = N.min_height), "max_height" in N && t(4, s = N.max_height), "width" in N && t(5, u = N.width), "elem_id" in N && t(6, h = N.elem_id), "elem_classes" in N && t(7, d = N.elem_classes), "variant" in N && t(8, g = N.variant), "border_mode" in N && t(9, p = N.border_mode), "padding" in N && t(10, v = N.padding), "type" in N && t(28, k = N.type), "test_id" in N && t(11, A = N.test_id), "explicit_call" in N && t(12, C = N.explicit_call), "container" in N && t(13, z = N.container), "visible" in N && t(14, x = N.visible), "allow_overflow" in N && t(15, _ = N.allow_overflow), "overflow_behavior" in N && t(16, w = N.overflow_behavior), "scale" in N && t(17, E = N.scale), "min_width" in N && t(18, T = N.min_width), "flex" in N && t(1, $ = N.flex), "resizable" in N && t(19, M = N.resizable), "rtl" in N && t(20, B = N.rtl), "fullscreen" in N && t(0, G = N.fullscreen), "$$scope" in N && t(30, a = N.$$scope);
  }, n.$$.update = () => {
    n.$$.dirty[0] & /*fullscreen, old_fullscreen, element*/
    538968065 && G !== U && (t(29, U = G), G ? (t(24, fe = j.getBoundingClientRect()), t(22, ee = j.offsetHeight), t(23, ue = j.offsetWidth), window.addEventListener("keydown", Ee)) : (t(24, fe = null), window.removeEventListener("keydown", Ee))), n.$$.dirty[0] & /*visible*/
    16384 && (x || t(1, $ = !1));
  }, [
    G,
    $,
    i,
    l,
    s,
    u,
    h,
    d,
    g,
    p,
    v,
    A,
    C,
    z,
    x,
    _,
    w,
    E,
    T,
    M,
    B,
    j,
    ee,
    ue,
    fe,
    oe,
    ne,
    ve,
    k,
    U,
    a,
    r,
    we
  ];
}
class Hu extends Eu {
  constructor(e) {
    super(), Bu(
      this,
      e,
      Pu,
      Ou,
      qu,
      {
        height: 2,
        min_height: 3,
        max_height: 4,
        width: 5,
        elem_id: 6,
        elem_classes: 7,
        variant: 8,
        border_mode: 9,
        padding: 10,
        type: 28,
        test_id: 11,
        explicit_call: 12,
        container: 13,
        visible: 14,
        allow_overflow: 15,
        overflow_behavior: 16,
        scale: 17,
        min_width: 18,
        flex: 1,
        resizable: 19,
        rtl: 20,
        fullscreen: 0
      },
      null,
      [-1, -1]
    );
  }
}
class Je {
  // The + prefix indicates that these fields aren't writeable
  // Lexer holding the input string.
  // Start offset, zero-based inclusive.
  // End offset, zero-based exclusive.
  constructor(e, t, r) {
    this.lexer = void 0, this.start = void 0, this.end = void 0, this.lexer = e, this.start = t, this.end = r;
  }
  /**
   * Merges two `SourceLocation`s from location providers, given they are
   * provided in order of appearance.
   * - Returns the first one's location if only the first is provided.
   * - Returns a merged range of the first and the last if both are provided
   *   and their lexers match.
   * - Otherwise, returns null.
   */
  static range(e, t) {
    return t ? !e || !e.loc || !t.loc || e.loc.lexer !== t.loc.lexer ? null : new Je(e.loc.lexer, e.loc.start, t.loc.end) : e && e.loc;
  }
}
class ot {
  // don't expand the token
  // used in \noexpand
  constructor(e, t) {
    this.text = void 0, this.loc = void 0, this.noexpand = void 0, this.treatAsRelax = void 0, this.text = e, this.loc = t;
  }
  /**
   * Given a pair of tokens (this and endToken), compute a `Token` encompassing
   * the whole input range enclosed by these two.
   */
  range(e, t) {
    return new ot(t, Je.range(this, e));
  }
}
class L {
  // Error start position based on passed-in Token or ParseNode.
  // Length of affected text based on passed-in Token or ParseNode.
  // The underlying error message without any context added.
  constructor(e, t) {
    this.name = void 0, this.position = void 0, this.length = void 0, this.rawMessage = void 0;
    var r = "KaTeX parse error: " + e, a, i, l = t && t.loc;
    if (l && l.start <= l.end) {
      var s = l.lexer.input;
      a = l.start, i = l.end, a === s.length ? r += " at end of input: " : r += " at position " + (a + 1) + ": ";
      var u = s.slice(a, i).replace(/[^]/g, "$&̲"), h;
      a > 15 ? h = "…" + s.slice(a - 15, a) : h = s.slice(0, a);
      var d;
      i + 15 < s.length ? d = s.slice(i, i + 15) + "…" : d = s.slice(i), r += h + u + d;
    }
    var g = new Error(r);
    return g.name = "ParseError", g.__proto__ = L.prototype, g.position = a, a != null && i != null && (g.length = i - a), g.rawMessage = e, g;
  }
}
L.prototype.__proto__ = Error.prototype;
var Uu = function(e, t) {
  return e.indexOf(t) !== -1;
}, Gu = function(e, t) {
  return e === void 0 ? t : e;
}, Vu = /([A-Z])/g, Wu = function(e) {
  return e.replace(Vu, "-$1").toLowerCase();
}, ju = {
  "&": "&amp;",
  ">": "&gt;",
  "<": "&lt;",
  '"': "&quot;",
  "'": "&#x27;"
}, Yu = /[&><"']/g;
function Xu(n) {
  return String(n).replace(Yu, (e) => ju[e]);
}
var Ds = function n(e) {
  return e.type === "ordgroup" || e.type === "color" ? e.body.length === 1 ? n(e.body[0]) : e : e.type === "font" ? n(e.body) : e;
}, Zu = function(e) {
  var t = Ds(e);
  return t.type === "mathord" || t.type === "textord" || t.type === "atom";
}, Ku = function(e) {
  if (!e)
    throw new Error("Expected non-null, but got " + String(e));
  return e;
}, Qu = function(e) {
  var t = /^[\x00-\x20]*([^\\/#?]*?)(:|&#0*58|&#x0*3a|&colon)/i.exec(e);
  return t ? t[2] !== ":" || !/^[a-zA-Z][a-zA-Z0-9+\-.]*$/.test(t[1]) ? null : t[1].toLowerCase() : "_relative";
}, Z = {
  contains: Uu,
  deflt: Gu,
  escape: Xu,
  hyphenate: Wu,
  getBaseElem: Ds,
  isCharacterBox: Zu,
  protocolFromUrl: Qu
}, er = {
  displayMode: {
    type: "boolean",
    description: "Render math in display mode, which puts the math in display style (so \\int and \\sum are large, for example), and centers the math on the page on its own line.",
    cli: "-d, --display-mode"
  },
  output: {
    type: {
      enum: ["htmlAndMathml", "html", "mathml"]
    },
    description: "Determines the markup language of the output.",
    cli: "-F, --format <type>"
  },
  leqno: {
    type: "boolean",
    description: "Render display math in leqno style (left-justified tags)."
  },
  fleqn: {
    type: "boolean",
    description: "Render display math flush left."
  },
  throwOnError: {
    type: "boolean",
    default: !0,
    cli: "-t, --no-throw-on-error",
    cliDescription: "Render errors (in the color given by --error-color) instead of throwing a ParseError exception when encountering an error."
  },
  errorColor: {
    type: "string",
    default: "#cc0000",
    cli: "-c, --error-color <color>",
    cliDescription: "A color string given in the format 'rgb' or 'rrggbb' (no #). This option determines the color of errors rendered by the -t option.",
    cliProcessor: (n) => "#" + n
  },
  macros: {
    type: "object",
    cli: "-m, --macro <def>",
    cliDescription: "Define custom macro of the form '\\foo:expansion' (use multiple -m arguments for multiple macros).",
    cliDefault: [],
    cliProcessor: (n, e) => (e.push(n), e)
  },
  minRuleThickness: {
    type: "number",
    description: "Specifies a minimum thickness, in ems, for fraction lines, `\\sqrt` top lines, `{array}` vertical lines, `\\hline`, `\\hdashline`, `\\underline`, `\\overline`, and the borders of `\\fbox`, `\\boxed`, and `\\fcolorbox`.",
    processor: (n) => Math.max(0, n),
    cli: "--min-rule-thickness <size>",
    cliProcessor: parseFloat
  },
  colorIsTextColor: {
    type: "boolean",
    description: "Makes \\color behave like LaTeX's 2-argument \\textcolor, instead of LaTeX's one-argument \\color mode change.",
    cli: "-b, --color-is-text-color"
  },
  strict: {
    type: [{
      enum: ["warn", "ignore", "error"]
    }, "boolean", "function"],
    description: "Turn on strict / LaTeX faithfulness mode, which throws an error if the input uses features that are not supported by LaTeX.",
    cli: "-S, --strict",
    cliDefault: !1
  },
  trust: {
    type: ["boolean", "function"],
    description: "Trust the input, enabling all HTML features such as \\url.",
    cli: "-T, --trust"
  },
  maxSize: {
    type: "number",
    default: 1 / 0,
    description: "If non-zero, all user-specified sizes, e.g. in \\rule{500em}{500em}, will be capped to maxSize ems. Otherwise, elements and spaces can be arbitrarily large",
    processor: (n) => Math.max(0, n),
    cli: "-s, --max-size <n>",
    cliProcessor: parseInt
  },
  maxExpand: {
    type: "number",
    default: 1e3,
    description: "Limit the number of macro expansions to the specified number, to prevent e.g. infinite macro loops. If set to Infinity, the macro expander will try to fully expand as in LaTeX.",
    processor: (n) => Math.max(0, n),
    cli: "-e, --max-expand <n>",
    cliProcessor: (n) => n === "Infinity" ? 1 / 0 : parseInt(n)
  },
  globalGroup: {
    type: "boolean",
    cli: !1
  }
};
function Ju(n) {
  if (n.default)
    return n.default;
  var e = n.type, t = Array.isArray(e) ? e[0] : e;
  if (typeof t != "string")
    return t.enum[0];
  switch (t) {
    case "boolean":
      return !1;
    case "string":
      return "";
    case "number":
      return 0;
    case "object":
      return {};
  }
}
class ka {
  constructor(e) {
    this.displayMode = void 0, this.output = void 0, this.leqno = void 0, this.fleqn = void 0, this.throwOnError = void 0, this.errorColor = void 0, this.macros = void 0, this.minRuleThickness = void 0, this.colorIsTextColor = void 0, this.strict = void 0, this.trust = void 0, this.maxSize = void 0, this.maxExpand = void 0, this.globalGroup = void 0, e = e || {};
    for (var t in er)
      if (er.hasOwnProperty(t)) {
        var r = er[t];
        this[t] = e[t] !== void 0 ? r.processor ? r.processor(e[t]) : e[t] : Ju(r);
      }
  }
  /**
   * Report nonstrict (non-LaTeX-compatible) input.
   * Can safely not be called if `this.strict` is false in JavaScript.
   */
  reportNonstrict(e, t, r) {
    var a = this.strict;
    if (typeof a == "function" && (a = a(e, t, r)), !(!a || a === "ignore")) {
      if (a === !0 || a === "error")
        throw new L("LaTeX-incompatible input and strict mode is set to 'error': " + (t + " [" + e + "]"), r);
      a === "warn" ? typeof console < "u" && console.warn("LaTeX-incompatible input and strict mode is set to 'warn': " + (t + " [" + e + "]")) : typeof console < "u" && console.warn("LaTeX-incompatible input and strict mode is set to " + ("unrecognized '" + a + "': " + t + " [" + e + "]"));
    }
  }
  /**
   * Check whether to apply strict (LaTeX-adhering) behavior for unusual
   * input (like `\\`).  Unlike `nonstrict`, will not throw an error;
   * instead, "error" translates to a return value of `true`, while "ignore"
   * translates to a return value of `false`.  May still print a warning:
   * "warn" prints a warning and returns `false`.
   * This is for the second category of `errorCode`s listed in the README.
   */
  useStrictBehavior(e, t, r) {
    var a = this.strict;
    if (typeof a == "function")
      try {
        a = a(e, t, r);
      } catch {
        a = "error";
      }
    return !a || a === "ignore" ? !1 : a === !0 || a === "error" ? !0 : a === "warn" ? (typeof console < "u" && console.warn("LaTeX-incompatible input and strict mode is set to 'warn': " + (t + " [" + e + "]")), !1) : (typeof console < "u" && console.warn("LaTeX-incompatible input and strict mode is set to " + ("unrecognized '" + a + "': " + t + " [" + e + "]")), !1);
  }
  /**
   * Check whether to test potentially dangerous input, and return
   * `true` (trusted) or `false` (untrusted).  The sole argument `context`
   * should be an object with `command` field specifying the relevant LaTeX
   * command (as a string starting with `\`), and any other arguments, etc.
   * If `context` has a `url` field, a `protocol` field will automatically
   * get added by this function (changing the specified object).
   */
  isTrusted(e) {
    if (e.url && !e.protocol) {
      var t = Z.protocolFromUrl(e.url);
      if (t == null)
        return !1;
      e.protocol = t;
    }
    var r = typeof this.trust == "function" ? this.trust(e) : this.trust;
    return !!r;
  }
}
class l0 {
  constructor(e, t, r) {
    this.id = void 0, this.size = void 0, this.cramped = void 0, this.id = e, this.size = t, this.cramped = r;
  }
  /**
   * Get the style of a superscript given a base in the current style.
   */
  sup() {
    return $t[e1[this.id]];
  }
  /**
   * Get the style of a subscript given a base in the current style.
   */
  sub() {
    return $t[t1[this.id]];
  }
  /**
   * Get the style of a fraction numerator given the fraction in the current
   * style.
   */
  fracNum() {
    return $t[r1[this.id]];
  }
  /**
   * Get the style of a fraction denominator given the fraction in the current
   * style.
   */
  fracDen() {
    return $t[n1[this.id]];
  }
  /**
   * Get the cramped version of a style (in particular, cramping a cramped style
   * doesn't change the style).
   */
  cramp() {
    return $t[a1[this.id]];
  }
  /**
   * Get a text or display version of this style.
   */
  text() {
    return $t[i1[this.id]];
  }
  /**
   * Return true if this style is tightly spaced (scriptstyle/scriptscriptstyle)
   */
  isTight() {
    return this.size >= 2;
  }
}
var Da = 0, Zr = 1, N0 = 2, e0 = 3, sr = 4, it = 5, L0 = 6, Ue = 7, $t = [new l0(Da, 0, !1), new l0(Zr, 0, !0), new l0(N0, 1, !1), new l0(e0, 1, !0), new l0(sr, 2, !1), new l0(it, 2, !0), new l0(L0, 3, !1), new l0(Ue, 3, !0)], e1 = [sr, it, sr, it, L0, Ue, L0, Ue], t1 = [it, it, it, it, Ue, Ue, Ue, Ue], r1 = [N0, e0, sr, it, L0, Ue, L0, Ue], n1 = [e0, e0, it, it, Ue, Ue, Ue, Ue], a1 = [Zr, Zr, e0, e0, it, it, Ue, Ue], i1 = [Da, Zr, N0, e0, N0, e0, N0, e0], Q = {
  DISPLAY: $t[Da],
  TEXT: $t[N0],
  SCRIPT: $t[sr],
  SCRIPTSCRIPT: $t[L0]
}, ea = [{
  // Latin characters beyond the Latin-1 characters we have metrics for.
  // Needed for Czech, Hungarian and Turkish text, for example.
  name: "latin",
  blocks: [
    [256, 591],
    // Latin Extended-A and Latin Extended-B
    [768, 879]
    // Combining Diacritical marks
  ]
}, {
  // The Cyrillic script used by Russian and related languages.
  // A Cyrillic subset used to be supported as explicitly defined
  // symbols in symbols.js
  name: "cyrillic",
  blocks: [[1024, 1279]]
}, {
  // Armenian
  name: "armenian",
  blocks: [[1328, 1423]]
}, {
  // The Brahmic scripts of South and Southeast Asia
  // Devanagari (0900–097F)
  // Bengali (0980–09FF)
  // Gurmukhi (0A00–0A7F)
  // Gujarati (0A80–0AFF)
  // Oriya (0B00–0B7F)
  // Tamil (0B80–0BFF)
  // Telugu (0C00–0C7F)
  // Kannada (0C80–0CFF)
  // Malayalam (0D00–0D7F)
  // Sinhala (0D80–0DFF)
  // Thai (0E00–0E7F)
  // Lao (0E80–0EFF)
  // Tibetan (0F00–0FFF)
  // Myanmar (1000–109F)
  name: "brahmic",
  blocks: [[2304, 4255]]
}, {
  name: "georgian",
  blocks: [[4256, 4351]]
}, {
  // Chinese and Japanese.
  // The "k" in cjk is for Korean, but we've separated Korean out
  name: "cjk",
  blocks: [
    [12288, 12543],
    // CJK symbols and punctuation, Hiragana, Katakana
    [19968, 40879],
    // CJK ideograms
    [65280, 65376]
    // Fullwidth punctuation
    // TODO: add halfwidth Katakana and Romanji glyphs
  ]
}, {
  // Korean
  name: "hangul",
  blocks: [[44032, 55215]]
}];
function l1(n) {
  for (var e = 0; e < ea.length; e++)
    for (var t = ea[e], r = 0; r < t.blocks.length; r++) {
      var a = t.blocks[r];
      if (n >= a[0] && n <= a[1])
        return t.name;
    }
  return null;
}
var Hr = [];
ea.forEach((n) => n.blocks.forEach((e) => Hr.push(...e)));
function Ss(n) {
  for (var e = 0; e < Hr.length; e += 2)
    if (n >= Hr[e] && n <= Hr[e + 1])
      return !0;
  return !1;
}
var $0 = 80, s1 = function(e, t) {
  return "M95," + (622 + e + t) + `
c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
c69,-144,104.5,-217.7,106.5,-221
l` + e / 2.075 + " -" + e + `
c5.3,-9.3,12,-14,20,-14
H400000v` + (40 + e) + `H845.2724
s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
M` + (834 + e) + " " + t + "h400000v" + (40 + e) + "h-400000z";
}, o1 = function(e, t) {
  return "M263," + (601 + e + t) + `c0.7,0,18,39.7,52,119
c34,79.3,68.167,158.7,102.5,238c34.3,79.3,51.8,119.3,52.5,120
c340,-704.7,510.7,-1060.3,512,-1067
l` + e / 2.084 + " -" + e + `
c4.7,-7.3,11,-11,19,-11
H40000v` + (40 + e) + `H1012.3
s-271.3,567,-271.3,567c-38.7,80.7,-84,175,-136,283c-52,108,-89.167,185.3,-111.5,232
c-22.3,46.7,-33.8,70.3,-34.5,71c-4.7,4.7,-12.3,7,-23,7s-12,-1,-12,-1
s-109,-253,-109,-253c-72.7,-168,-109.3,-252,-110,-252c-10.7,8,-22,16.7,-34,26
c-22,17.3,-33.3,26,-34,26s-26,-26,-26,-26s76,-59,76,-59s76,-60,76,-60z
M` + (1001 + e) + " " + t + "h400000v" + (40 + e) + "h-400000z";
}, u1 = function(e, t) {
  return "M983 " + (10 + e + t) + `
l` + e / 3.13 + " -" + e + `
c4,-6.7,10,-10,18,-10 H400000v` + (40 + e) + `
H1013.1s-83.4,268,-264.1,840c-180.7,572,-277,876.3,-289,913c-4.7,4.7,-12.7,7,-24,7
s-12,0,-12,0c-1.3,-3.3,-3.7,-11.7,-7,-25c-35.3,-125.3,-106.7,-373.3,-214,-744
c-10,12,-21,25,-33,39s-32,39,-32,39c-6,-5.3,-15,-14,-27,-26s25,-30,25,-30
c26.7,-32.7,52,-63,76,-91s52,-60,52,-60s208,722,208,722
c56,-175.3,126.3,-397.3,211,-666c84.7,-268.7,153.8,-488.2,207.5,-658.5
c53.7,-170.3,84.5,-266.8,92.5,-289.5z
M` + (1001 + e) + " " + t + "h400000v" + (40 + e) + "h-400000z";
}, c1 = function(e, t) {
  return "M424," + (2398 + e + t) + `
c-1.3,-0.7,-38.5,-172,-111.5,-514c-73,-342,-109.8,-513.3,-110.5,-514
c0,-2,-10.7,14.3,-32,49c-4.7,7.3,-9.8,15.7,-15.5,25c-5.7,9.3,-9.8,16,-12.5,20
s-5,7,-5,7c-4,-3.3,-8.3,-7.7,-13,-13s-13,-13,-13,-13s76,-122,76,-122s77,-121,77,-121
s209,968,209,968c0,-2,84.7,-361.7,254,-1079c169.3,-717.3,254.7,-1077.7,256,-1081
l` + e / 4.223 + " -" + e + `c4,-6.7,10,-10,18,-10 H400000
v` + (40 + e) + `H1014.6
s-87.3,378.7,-272.6,1166c-185.3,787.3,-279.3,1182.3,-282,1185
c-2,6,-10,9,-24,9
c-8,0,-12,-0.7,-12,-2z M` + (1001 + e) + " " + t + `
h400000v` + (40 + e) + "h-400000z";
}, h1 = function(e, t) {
  return "M473," + (2713 + e + t) + `
c339.3,-1799.3,509.3,-2700,510,-2702 l` + e / 5.298 + " -" + e + `
c3.3,-7.3,9.3,-11,18,-11 H400000v` + (40 + e) + `H1017.7
s-90.5,478,-276.2,1466c-185.7,988,-279.5,1483,-281.5,1485c-2,6,-10,9,-24,9
c-8,0,-12,-0.7,-12,-2c0,-1.3,-5.3,-32,-16,-92c-50.7,-293.3,-119.7,-693.3,-207,-1200
c0,-1.3,-5.3,8.7,-16,30c-10.7,21.3,-21.3,42.7,-32,64s-16,33,-16,33s-26,-26,-26,-26
s76,-153,76,-153s77,-151,77,-151c0.7,0.7,35.7,202,105,604c67.3,400.7,102,602.7,104,
606zM` + (1001 + e) + " " + t + "h400000v" + (40 + e) + "H1017.7z";
}, d1 = function(e) {
  var t = e / 2;
  return "M400000 " + e + " H0 L" + t + " 0 l65 45 L145 " + (e - 80) + " H400000z";
}, m1 = function(e, t, r) {
  var a = r - 54 - t - e;
  return "M702 " + (e + t) + "H400000" + (40 + e) + `
H742v` + a + `l-4 4-4 4c-.667.7 -2 1.5-4 2.5s-4.167 1.833-6.5 2.5-5.5 1-9.5 1
h-12l-28-84c-16.667-52-96.667 -294.333-240-727l-212 -643 -85 170
c-4-3.333-8.333-7.667-13 -13l-13-13l77-155 77-156c66 199.333 139 419.667
219 661 l218 661zM702 ` + t + "H400000v" + (40 + e) + "H742z";
}, f1 = function(e, t, r) {
  t = 1e3 * t;
  var a = "";
  switch (e) {
    case "sqrtMain":
      a = s1(t, $0);
      break;
    case "sqrtSize1":
      a = o1(t, $0);
      break;
    case "sqrtSize2":
      a = u1(t, $0);
      break;
    case "sqrtSize3":
      a = c1(t, $0);
      break;
    case "sqrtSize4":
      a = h1(t, $0);
      break;
    case "sqrtTall":
      a = m1(t, $0, r);
  }
  return a;
}, p1 = function(e, t) {
  switch (e) {
    case "⎜":
      return "M291 0 H417 V" + t + " H291z M291 0 H417 V" + t + " H291z";
    case "∣":
      return "M145 0 H188 V" + t + " H145z M145 0 H188 V" + t + " H145z";
    case "∥":
      return "M145 0 H188 V" + t + " H145z M145 0 H188 V" + t + " H145z" + ("M367 0 H410 V" + t + " H367z M367 0 H410 V" + t + " H367z");
    case "⎟":
      return "M457 0 H583 V" + t + " H457z M457 0 H583 V" + t + " H457z";
    case "⎢":
      return "M319 0 H403 V" + t + " H319z M319 0 H403 V" + t + " H319z";
    case "⎥":
      return "M263 0 H347 V" + t + " H263z M263 0 H347 V" + t + " H263z";
    case "⎪":
      return "M384 0 H504 V" + t + " H384z M384 0 H504 V" + t + " H384z";
    case "⏐":
      return "M312 0 H355 V" + t + " H312z M312 0 H355 V" + t + " H312z";
    case "‖":
      return "M257 0 H300 V" + t + " H257z M257 0 H300 V" + t + " H257z" + ("M478 0 H521 V" + t + " H478z M478 0 H521 V" + t + " H478z");
    default:
      return "";
  }
}, Si = {
  // The doubleleftarrow geometry is from glyph U+21D0 in the font KaTeX Main
  doubleleftarrow: `M262 157
l10-10c34-36 62.7-77 86-123 3.3-8 5-13.3 5-16 0-5.3-6.7-8-20-8-7.3
 0-12.2.5-14.5 1.5-2.3 1-4.8 4.5-7.5 10.5-49.3 97.3-121.7 169.3-217 216-28
 14-57.3 25-88 33-6.7 2-11 3.8-13 5.5-2 1.7-3 4.2-3 7.5s1 5.8 3 7.5
c2 1.7 6.3 3.5 13 5.5 68 17.3 128.2 47.8 180.5 91.5 52.3 43.7 93.8 96.2 124.5
 157.5 9.3 8 15.3 12.3 18 13h6c12-.7 18-4 18-10 0-2-1.7-7-5-15-23.3-46-52-87
-86-123l-10-10h399738v-40H218c328 0 0 0 0 0l-10-8c-26.7-20-65.7-43-117-69 2.7
-2 6-3.7 10-5 36.7-16 72.3-37.3 107-64l10-8h399782v-40z
m8 0v40h399730v-40zm0 194v40h399730v-40z`,
  // doublerightarrow is from glyph U+21D2 in font KaTeX Main
  doublerightarrow: `M399738 392l
-10 10c-34 36-62.7 77-86 123-3.3 8-5 13.3-5 16 0 5.3 6.7 8 20 8 7.3 0 12.2-.5
 14.5-1.5 2.3-1 4.8-4.5 7.5-10.5 49.3-97.3 121.7-169.3 217-216 28-14 57.3-25 88
-33 6.7-2 11-3.8 13-5.5 2-1.7 3-4.2 3-7.5s-1-5.8-3-7.5c-2-1.7-6.3-3.5-13-5.5-68
-17.3-128.2-47.8-180.5-91.5-52.3-43.7-93.8-96.2-124.5-157.5-9.3-8-15.3-12.3-18
-13h-6c-12 .7-18 4-18 10 0 2 1.7 7 5 15 23.3 46 52 87 86 123l10 10H0v40h399782
c-328 0 0 0 0 0l10 8c26.7 20 65.7 43 117 69-2.7 2-6 3.7-10 5-36.7 16-72.3 37.3
-107 64l-10 8H0v40zM0 157v40h399730v-40zm0 194v40h399730v-40z`,
  // leftarrow is from glyph U+2190 in font KaTeX Main
  leftarrow: `M400000 241H110l3-3c68.7-52.7 113.7-120
 135-202 4-14.7 6-23 6-25 0-7.3-7-11-21-11-8 0-13.2.8-15.5 2.5-2.3 1.7-4.2 5.8
-5.5 12.5-1.3 4.7-2.7 10.3-4 17-12 48.7-34.8 92-68.5 130S65.3 228.3 18 247
c-10 4-16 7.7-18 11 0 8.7 6 14.3 18 17 47.3 18.7 87.8 47 121.5 85S196 441.3 208
 490c.7 2 1.3 5 2 9s1.2 6.7 1.5 8c.3 1.3 1 3.3 2 6s2.2 4.5 3.5 5.5c1.3 1 3.3
 1.8 6 2.5s6 1 10 1c14 0 21-3.7 21-11 0-2-2-10.3-6-25-20-79.3-65-146.7-135-202
 l-3-3h399890zM100 241v40h399900v-40z`,
  // overbrace is from glyphs U+23A9/23A8/23A7 in font KaTeX_Size4-Regular
  leftbrace: `M6 548l-6-6v-35l6-11c56-104 135.3-181.3 238-232 57.3-28.7 117
-45 179-50h399577v120H403c-43.3 7-81 15-113 26-100.7 33-179.7 91-237 174-2.7
 5-6 9-10 13-.7 1-7.3 1-20 1H6z`,
  leftbraceunder: `M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13
 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688
 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7
-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z`,
  // overgroup is from the MnSymbol package (public domain)
  leftgroup: `M400000 80
H435C64 80 168.3 229.4 21 260c-5.9 1.2-18 0-18 0-2 0-3-1-3-3v-38C76 61 257 0
 435 0h399565z`,
  leftgroupunder: `M400000 262
H435C64 262 168.3 112.6 21 82c-5.9-1.2-18 0-18 0-2 0-3 1-3 3v38c76 158 257 219
 435 219h399565z`,
  // Harpoons are from glyph U+21BD in font KaTeX Main
  leftharpoon: `M0 267c.7 5.3 3 10 7 14h399993v-40H93c3.3
-3.3 10.2-9.5 20.5-18.5s17.8-15.8 22.5-20.5c50.7-52 88-110.3 112-175 4-11.3 5
-18.3 3-21-1.3-4-7.3-6-18-6-8 0-13 .7-15 2s-4.7 6.7-8 16c-42 98.7-107.3 174.7
-196 228-6.7 4.7-10.7 8-12 10-1.3 2-2 5.7-2 11zm100-26v40h399900v-40z`,
  leftharpoonplus: `M0 267c.7 5.3 3 10 7 14h399993v-40H93c3.3-3.3 10.2-9.5
 20.5-18.5s17.8-15.8 22.5-20.5c50.7-52 88-110.3 112-175 4-11.3 5-18.3 3-21-1.3
-4-7.3-6-18-6-8 0-13 .7-15 2s-4.7 6.7-8 16c-42 98.7-107.3 174.7-196 228-6.7 4.7
-10.7 8-12 10-1.3 2-2 5.7-2 11zm100-26v40h399900v-40zM0 435v40h400000v-40z
m0 0v40h400000v-40z`,
  leftharpoondown: `M7 241c-4 4-6.333 8.667-7 14 0 5.333.667 9 2 11s5.333
 5.333 12 10c90.667 54 156 130 196 228 3.333 10.667 6.333 16.333 9 17 2 .667 5
 1 9 1h5c10.667 0 16.667-2 18-6 2-2.667 1-9.667-3-21-32-87.333-82.667-157.667
-152-211l-3-3h399907v-40zM93 281 H400000 v-40L7 241z`,
  leftharpoondownplus: `M7 435c-4 4-6.3 8.7-7 14 0 5.3.7 9 2 11s5.3 5.3 12
 10c90.7 54 156 130 196 228 3.3 10.7 6.3 16.3 9 17 2 .7 5 1 9 1h5c10.7 0 16.7
-2 18-6 2-2.7 1-9.7-3-21-32-87.3-82.7-157.7-152-211l-3-3h399907v-40H7zm93 0
v40h399900v-40zM0 241v40h399900v-40zm0 0v40h399900v-40z`,
  // hook is from glyph U+21A9 in font KaTeX Main
  lefthook: `M400000 281 H103s-33-11.2-61-33.5S0 197.3 0 164s14.2-61.2 42.5
-83.5C70.8 58.2 104 47 142 47 c16.7 0 25 6.7 25 20 0 12-8.7 18.7-26 20-40 3.3
-68.7 15.7-86 37-10 12-15 25.3-15 40 0 22.7 9.8 40.7 29.5 54 19.7 13.3 43.5 21
 71.5 23h399859zM103 281v-40h399897v40z`,
  leftlinesegment: `M40 281 V428 H0 V94 H40 V241 H400000 v40z
M40 281 V428 H0 V94 H40 V241 H400000 v40z`,
  leftmapsto: `M40 281 V448H0V74H40V241H400000v40z
M40 281 V448H0V74H40V241H400000v40z`,
  // tofrom is from glyph U+21C4 in font KaTeX AMS Regular
  leftToFrom: `M0 147h400000v40H0zm0 214c68 40 115.7 95.7 143 167h22c15.3 0 23
-.3 23-1 0-1.3-5.3-13.7-16-37-18-35.3-41.3-69-70-101l-7-8h399905v-40H95l7-8
c28.7-32 52-65.7 70-101 10.7-23.3 16-35.7 16-37 0-.7-7.7-1-23-1h-22C115.7 265.3
 68 321 0 361zm0-174v-40h399900v40zm100 154v40h399900v-40z`,
  longequal: `M0 50 h400000 v40H0z m0 194h40000v40H0z
M0 50 h400000 v40H0z m0 194h40000v40H0z`,
  midbrace: `M200428 334
c-100.7-8.3-195.3-44-280-108-55.3-42-101.7-93-139-153l-9-14c-2.7 4-5.7 8.7-9 14
-53.3 86.7-123.7 153-211 199-66.7 36-137.3 56.3-212 62H0V214h199568c178.3-11.7
 311.7-78.3 403-201 6-8 9.7-12 11-12 .7-.7 6.7-1 18-1s17.3.3 18 1c1.3 0 5 4 11
 12 44.7 59.3 101.3 106.3 170 141s145.3 54.3 229 60h199572v120z`,
  midbraceunder: `M199572 214
c100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14
 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3
 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0
-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z`,
  oiintSize1: `M512.6 71.6c272.6 0 320.3 106.8 320.3 178.2 0 70.8-47.7 177.6
-320.3 177.6S193.1 320.6 193.1 249.8c0-71.4 46.9-178.2 319.5-178.2z
m368.1 178.2c0-86.4-60.9-215.4-368.1-215.4-306.4 0-367.3 129-367.3 215.4 0 85.8
60.9 214.8 367.3 214.8 307.2 0 368.1-129 368.1-214.8z`,
  oiintSize2: `M757.8 100.1c384.7 0 451.1 137.6 451.1 230 0 91.3-66.4 228.8
-451.1 228.8-386.3 0-452.7-137.5-452.7-228.8 0-92.4 66.4-230 452.7-230z
m502.4 230c0-111.2-82.4-277.2-502.4-277.2s-504 166-504 277.2
c0 110 84 276 504 276s502.4-166 502.4-276z`,
  oiiintSize1: `M681.4 71.6c408.9 0 480.5 106.8 480.5 178.2 0 70.8-71.6 177.6
-480.5 177.6S202.1 320.6 202.1 249.8c0-71.4 70.5-178.2 479.3-178.2z
m525.8 178.2c0-86.4-86.8-215.4-525.7-215.4-437.9 0-524.7 129-524.7 215.4 0
85.8 86.8 214.8 524.7 214.8 438.9 0 525.7-129 525.7-214.8z`,
  oiiintSize2: `M1021.2 53c603.6 0 707.8 165.8 707.8 277.2 0 110-104.2 275.8
-707.8 275.8-606 0-710.2-165.8-710.2-275.8C311 218.8 415.2 53 1021.2 53z
m770.4 277.1c0-131.2-126.4-327.6-770.5-327.6S248.4 198.9 248.4 330.1
c0 130 128.8 326.4 772.7 326.4s770.5-196.4 770.5-326.4z`,
  rightarrow: `M0 241v40h399891c-47.3 35.3-84 78-110 128
-16.7 32-27.7 63.7-33 95 0 1.3-.2 2.7-.5 4-.3 1.3-.5 2.3-.5 3 0 7.3 6.7 11 20
 11 8 0 13.2-.8 15.5-2.5 2.3-1.7 4.2-5.5 5.5-11.5 2-13.3 5.7-27 11-41 14.7-44.7
 39-84.5 73-119.5s73.7-60.2 119-75.5c6-2 9-5.7 9-11s-3-9-9-11c-45.3-15.3-85
-40.5-119-75.5s-58.3-74.8-73-119.5c-4.7-14-8.3-27.3-11-40-1.3-6.7-3.2-10.8-5.5
-12.5-2.3-1.7-7.5-2.5-15.5-2.5-14 0-21 3.7-21 11 0 2 2 10.3 6 25 20.7 83.3 67
 151.7 139 205zm0 0v40h399900v-40z`,
  rightbrace: `M400000 542l
-6 6h-17c-12.7 0-19.3-.3-20-1-4-4-7.3-8.3-10-13-35.3-51.3-80.8-93.8-136.5-127.5
s-117.2-55.8-184.5-66.5c-.7 0-2-.3-4-1-18.7-2.7-76-4.3-172-5H0V214h399571l6 1
c124.7 8 235 61.7 331 161 31.3 33.3 59.7 72.7 85 118l7 13v35z`,
  rightbraceunder: `M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3
 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237
-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z`,
  rightgroup: `M0 80h399565c371 0 266.7 149.4 414 180 5.9 1.2 18 0 18 0 2 0
 3-1 3-3v-38c-76-158-257-219-435-219H0z`,
  rightgroupunder: `M0 262h399565c371 0 266.7-149.4 414-180 5.9-1.2 18 0 18
 0 2 0 3 1 3 3v38c-76 158-257 219-435 219H0z`,
  rightharpoon: `M0 241v40h399993c4.7-4.7 7-9.3 7-14 0-9.3
-3.7-15.3-11-18-92.7-56.7-159-133.7-199-231-3.3-9.3-6-14.7-8-16-2-1.3-7-2-15-2
-10.7 0-16.7 2-18 6-2 2.7-1 9.7 3 21 15.3 42 36.7 81.8 64 119.5 27.3 37.7 58
 69.2 92 94.5zm0 0v40h399900v-40z`,
  rightharpoonplus: `M0 241v40h399993c4.7-4.7 7-9.3 7-14 0-9.3-3.7-15.3-11
-18-92.7-56.7-159-133.7-199-231-3.3-9.3-6-14.7-8-16-2-1.3-7-2-15-2-10.7 0-16.7
 2-18 6-2 2.7-1 9.7 3 21 15.3 42 36.7 81.8 64 119.5 27.3 37.7 58 69.2 92 94.5z
m0 0v40h399900v-40z m100 194v40h399900v-40zm0 0v40h399900v-40z`,
  rightharpoondown: `M399747 511c0 7.3 6.7 11 20 11 8 0 13-.8 15-2.5s4.7-6.8
 8-15.5c40-94 99.3-166.3 178-217 13.3-8 20.3-12.3 21-13 5.3-3.3 8.5-5.8 9.5
-7.5 1-1.7 1.5-5.2 1.5-10.5s-2.3-10.3-7-15H0v40h399908c-34 25.3-64.7 57-92 95
-27.3 38-48.7 77.7-64 119-3.3 8.7-5 14-5 16zM0 241v40h399900v-40z`,
  rightharpoondownplus: `M399747 705c0 7.3 6.7 11 20 11 8 0 13-.8
 15-2.5s4.7-6.8 8-15.5c40-94 99.3-166.3 178-217 13.3-8 20.3-12.3 21-13 5.3-3.3
 8.5-5.8 9.5-7.5 1-1.7 1.5-5.2 1.5-10.5s-2.3-10.3-7-15H0v40h399908c-34 25.3
-64.7 57-92 95-27.3 38-48.7 77.7-64 119-3.3 8.7-5 14-5 16zM0 435v40h399900v-40z
m0-194v40h400000v-40zm0 0v40h400000v-40z`,
  righthook: `M399859 241c-764 0 0 0 0 0 40-3.3 68.7-15.7 86-37 10-12 15-25.3
 15-40 0-22.7-9.8-40.7-29.5-54-19.7-13.3-43.5-21-71.5-23-17.3-1.3-26-8-26-20 0
-13.3 8.7-20 26-20 38 0 71 11.2 99 33.5 0 0 7 5.6 21 16.7 14 11.2 21 33.5 21
 66.8s-14 61.2-42 83.5c-28 22.3-61 33.5-99 33.5L0 241z M0 281v-40h399859v40z`,
  rightlinesegment: `M399960 241 V94 h40 V428 h-40 V281 H0 v-40z
M399960 241 V94 h40 V428 h-40 V281 H0 v-40z`,
  rightToFrom: `M400000 167c-70.7-42-118-97.7-142-167h-23c-15.3 0-23 .3-23
 1 0 1.3 5.3 13.7 16 37 18 35.3 41.3 69 70 101l7 8H0v40h399905l-7 8c-28.7 32
-52 65.7-70 101-10.7 23.3-16 35.7-16 37 0 .7 7.7 1 23 1h23c24-69.3 71.3-125 142
-167z M100 147v40h399900v-40zM0 341v40h399900v-40z`,
  // twoheadleftarrow is from glyph U+219E in font KaTeX AMS Regular
  twoheadleftarrow: `M0 167c68 40
 115.7 95.7 143 167h22c15.3 0 23-.3 23-1 0-1.3-5.3-13.7-16-37-18-35.3-41.3-69
-70-101l-7-8h125l9 7c50.7 39.3 85 86 103 140h46c0-4.7-6.3-18.7-19-42-18-35.3
-40-67.3-66-96l-9-9h399716v-40H284l9-9c26-28.7 48-60.7 66-96 12.7-23.333 19
-37.333 19-42h-46c-18 54-52.3 100.7-103 140l-9 7H95l7-8c28.7-32 52-65.7 70-101
 10.7-23.333 16-35.7 16-37 0-.7-7.7-1-23-1h-22C115.7 71.3 68 127 0 167z`,
  twoheadrightarrow: `M400000 167
c-68-40-115.7-95.7-143-167h-22c-15.3 0-23 .3-23 1 0 1.3 5.3 13.7 16 37 18 35.3
 41.3 69 70 101l7 8h-125l-9-7c-50.7-39.3-85-86-103-140h-46c0 4.7 6.3 18.7 19 42
 18 35.3 40 67.3 66 96l9 9H0v40h399716l-9 9c-26 28.7-48 60.7-66 96-12.7 23.333
-19 37.333-19 42h46c18-54 52.3-100.7 103-140l9-7h125l-7 8c-28.7 32-52 65.7-70
 101-10.7 23.333-16 35.7-16 37 0 .7 7.7 1 23 1h22c27.3-71.3 75-127 143-167z`,
  // tilde1 is a modified version of a glyph from the MnSymbol package
  tilde1: `M200 55.538c-77 0-168 73.953-177 73.953-3 0-7
-2.175-9-5.437L2 97c-1-2-2-4-2-6 0-4 2-7 5-9l20-12C116 12 171 0 207 0c86 0
 114 68 191 68 78 0 168-68 177-68 4 0 7 2 9 5l12 19c1 2.175 2 4.35 2 6.525 0
 4.35-2 7.613-5 9.788l-19 13.05c-92 63.077-116.937 75.308-183 76.128
-68.267.847-113-73.952-191-73.952z`,
  // ditto tilde2, tilde3, & tilde4
  tilde2: `M344 55.266c-142 0-300.638 81.316-311.5 86.418
-8.01 3.762-22.5 10.91-23.5 5.562L1 120c-1-2-1-3-1-4 0-5 3-9 8-10l18.4-9C160.9
 31.9 283 0 358 0c148 0 188 122 331 122s314-97 326-97c4 0 8 2 10 7l7 21.114
c1 2.14 1 3.21 1 4.28 0 5.347-3 9.626-7 10.696l-22.3 12.622C852.6 158.372 751
 181.476 676 181.476c-149 0-189-126.21-332-126.21z`,
  tilde3: `M786 59C457 59 32 175.242 13 175.242c-6 0-10-3.457
-11-10.37L.15 138c-1-7 3-12 10-13l19.2-6.4C378.4 40.7 634.3 0 804.3 0c337 0
 411.8 157 746.8 157 328 0 754-112 773-112 5 0 10 3 11 9l1 14.075c1 8.066-.697
 16.595-6.697 17.492l-21.052 7.31c-367.9 98.146-609.15 122.696-778.15 122.696
 -338 0-409-156.573-744-156.573z`,
  tilde4: `M786 58C457 58 32 177.487 13 177.487c-6 0-10-3.345
-11-10.035L.15 143c-1-7 3-12 10-13l22-6.7C381.2 35 637.15 0 807.15 0c337 0 409
 177 744 177 328 0 754-127 773-127 5 0 10 3 11 9l1 14.794c1 7.805-3 13.38-9
 14.495l-20.7 5.574c-366.85 99.79-607.3 139.372-776.3 139.372-338 0-409
 -175.236-744-175.236z`,
  // vec is from glyph U+20D7 in font KaTeX Main
  vec: `M377 20c0-5.333 1.833-10 5.5-14S391 0 397 0c4.667 0 8.667 1.667 12 5
3.333 2.667 6.667 9 10 19 6.667 24.667 20.333 43.667 41 57 7.333 4.667 11
10.667 11 18 0 6-1 10-3 12s-6.667 5-14 9c-28.667 14.667-53.667 35.667-75 63
-1.333 1.333-3.167 3.5-5.5 6.5s-4 4.833-5 5.5c-1 .667-2.5 1.333-4.5 2s-4.333 1
-7 1c-4.667 0-9.167-1.833-13.5-5.5S337 184 337 178c0-12.667 15.667-32.333 47-59
H213l-171-1c-8.667-6-13-12.333-13-19 0-4.667 4.333-11.333 13-20h359
c-16-25.333-24-45-24-59z`,
  // widehat1 is a modified version of a glyph from the MnSymbol package
  widehat1: `M529 0h5l519 115c5 1 9 5 9 10 0 1-1 2-1 3l-4 22
c-1 5-5 9-11 9h-2L532 67 19 159h-2c-5 0-9-4-11-9l-5-22c-1-6 2-12 8-13z`,
  // ditto widehat2, widehat3, & widehat4
  widehat2: `M1181 0h2l1171 176c6 0 10 5 10 11l-2 23c-1 6-5 10
-11 10h-1L1182 67 15 220h-1c-6 0-10-4-11-10l-2-23c-1-6 4-11 10-11z`,
  widehat3: `M1181 0h2l1171 236c6 0 10 5 10 11l-2 23c-1 6-5 10
-11 10h-1L1182 67 15 280h-1c-6 0-10-4-11-10l-2-23c-1-6 4-11 10-11z`,
  widehat4: `M1181 0h2l1171 296c6 0 10 5 10 11l-2 23c-1 6-5 10
-11 10h-1L1182 67 15 340h-1c-6 0-10-4-11-10l-2-23c-1-6 4-11 10-11z`,
  // widecheck paths are all inverted versions of widehat
  widecheck1: `M529,159h5l519,-115c5,-1,9,-5,9,-10c0,-1,-1,-2,-1,-3l-4,-22c-1,
-5,-5,-9,-11,-9h-2l-512,92l-513,-92h-2c-5,0,-9,4,-11,9l-5,22c-1,6,2,12,8,13z`,
  widecheck2: `M1181,220h2l1171,-176c6,0,10,-5,10,-11l-2,-23c-1,-6,-5,-10,
-11,-10h-1l-1168,153l-1167,-153h-1c-6,0,-10,4,-11,10l-2,23c-1,6,4,11,10,11z`,
  widecheck3: `M1181,280h2l1171,-236c6,0,10,-5,10,-11l-2,-23c-1,-6,-5,-10,
-11,-10h-1l-1168,213l-1167,-213h-1c-6,0,-10,4,-11,10l-2,23c-1,6,4,11,10,11z`,
  widecheck4: `M1181,340h2l1171,-296c6,0,10,-5,10,-11l-2,-23c-1,-6,-5,-10,
-11,-10h-1l-1168,273l-1167,-273h-1c-6,0,-10,4,-11,10l-2,23c-1,6,4,11,10,11z`,
  // The next ten paths support reaction arrows from the mhchem package.
  // Arrows for \ce{<-->} are offset from xAxis by 0.22ex, per mhchem in LaTeX
  // baraboveleftarrow is mostly from glyph U+2190 in font KaTeX Main
  baraboveleftarrow: `M400000 620h-399890l3 -3c68.7 -52.7 113.7 -120 135 -202
c4 -14.7 6 -23 6 -25c0 -7.3 -7 -11 -21 -11c-8 0 -13.2 0.8 -15.5 2.5
c-2.3 1.7 -4.2 5.8 -5.5 12.5c-1.3 4.7 -2.7 10.3 -4 17c-12 48.7 -34.8 92 -68.5 130
s-74.2 66.3 -121.5 85c-10 4 -16 7.7 -18 11c0 8.7 6 14.3 18 17c47.3 18.7 87.8 47
121.5 85s56.5 81.3 68.5 130c0.7 2 1.3 5 2 9s1.2 6.7 1.5 8c0.3 1.3 1 3.3 2 6
s2.2 4.5 3.5 5.5c1.3 1 3.3 1.8 6 2.5s6 1 10 1c14 0 21 -3.7 21 -11
c0 -2 -2 -10.3 -6 -25c-20 -79.3 -65 -146.7 -135 -202l-3 -3h399890z
M100 620v40h399900v-40z M0 241v40h399900v-40zM0 241v40h399900v-40z`,
  // rightarrowabovebar is mostly from glyph U+2192, KaTeX Main
  rightarrowabovebar: `M0 241v40h399891c-47.3 35.3-84 78-110 128-16.7 32
-27.7 63.7-33 95 0 1.3-.2 2.7-.5 4-.3 1.3-.5 2.3-.5 3 0 7.3 6.7 11 20 11 8 0
13.2-.8 15.5-2.5 2.3-1.7 4.2-5.5 5.5-11.5 2-13.3 5.7-27 11-41 14.7-44.7 39
-84.5 73-119.5s73.7-60.2 119-75.5c6-2 9-5.7 9-11s-3-9-9-11c-45.3-15.3-85-40.5
-119-75.5s-58.3-74.8-73-119.5c-4.7-14-8.3-27.3-11-40-1.3-6.7-3.2-10.8-5.5
-12.5-2.3-1.7-7.5-2.5-15.5-2.5-14 0-21 3.7-21 11 0 2 2 10.3 6 25 20.7 83.3 67
151.7 139 205zm96 379h399894v40H0zm0 0h399904v40H0z`,
  // The short left harpoon has 0.5em (i.e. 500 units) kern on the left end.
  // Ref from mhchem.sty: \rlap{\raisebox{-.22ex}{$\kern0.5em
  baraboveshortleftharpoon: `M507,435c-4,4,-6.3,8.7,-7,14c0,5.3,0.7,9,2,11
c1.3,2,5.3,5.3,12,10c90.7,54,156,130,196,228c3.3,10.7,6.3,16.3,9,17
c2,0.7,5,1,9,1c0,0,5,0,5,0c10.7,0,16.7,-2,18,-6c2,-2.7,1,-9.7,-3,-21
c-32,-87.3,-82.7,-157.7,-152,-211c0,0,-3,-3,-3,-3l399351,0l0,-40
c-398570,0,-399437,0,-399437,0z M593 435 v40 H399500 v-40z
M0 281 v-40 H399908 v40z M0 281 v-40 H399908 v40z`,
  rightharpoonaboveshortbar: `M0,241 l0,40c399126,0,399993,0,399993,0
c4.7,-4.7,7,-9.3,7,-14c0,-9.3,-3.7,-15.3,-11,-18c-92.7,-56.7,-159,-133.7,-199,
-231c-3.3,-9.3,-6,-14.7,-8,-16c-2,-1.3,-7,-2,-15,-2c-10.7,0,-16.7,2,-18,6
c-2,2.7,-1,9.7,3,21c15.3,42,36.7,81.8,64,119.5c27.3,37.7,58,69.2,92,94.5z
M0 241 v40 H399908 v-40z M0 475 v-40 H399500 v40z M0 475 v-40 H399500 v40z`,
  shortbaraboveleftharpoon: `M7,435c-4,4,-6.3,8.7,-7,14c0,5.3,0.7,9,2,11
c1.3,2,5.3,5.3,12,10c90.7,54,156,130,196,228c3.3,10.7,6.3,16.3,9,17c2,0.7,5,1,9,
1c0,0,5,0,5,0c10.7,0,16.7,-2,18,-6c2,-2.7,1,-9.7,-3,-21c-32,-87.3,-82.7,-157.7,
-152,-211c0,0,-3,-3,-3,-3l399907,0l0,-40c-399126,0,-399993,0,-399993,0z
M93 435 v40 H400000 v-40z M500 241 v40 H400000 v-40z M500 241 v40 H400000 v-40z`,
  shortrightharpoonabovebar: `M53,241l0,40c398570,0,399437,0,399437,0
c4.7,-4.7,7,-9.3,7,-14c0,-9.3,-3.7,-15.3,-11,-18c-92.7,-56.7,-159,-133.7,-199,
-231c-3.3,-9.3,-6,-14.7,-8,-16c-2,-1.3,-7,-2,-15,-2c-10.7,0,-16.7,2,-18,6
c-2,2.7,-1,9.7,3,21c15.3,42,36.7,81.8,64,119.5c27.3,37.7,58,69.2,92,94.5z
M500 241 v40 H399408 v-40z M500 435 v40 H400000 v-40z`
}, g1 = function(e, t) {
  switch (e) {
    case "lbrack":
      return "M403 1759 V84 H666 V0 H319 V1759 v" + t + ` v1759 h347 v-84
H403z M403 1759 V0 H319 V1759 v` + t + " v1759 h84z";
    case "rbrack":
      return "M347 1759 V0 H0 V84 H263 V1759 v" + t + ` v1759 H0 v84 H347z
M347 1759 V0 H263 V1759 v` + t + " v1759 h84z";
    case "vert":
      return "M145 15 v585 v" + t + ` v585 c2.667,10,9.667,15,21,15
c10,0,16.667,-5,20,-15 v-585 v` + -t + ` v-585 c-2.667,-10,-9.667,-15,-21,-15
c-10,0,-16.667,5,-20,15z M188 15 H145 v585 v` + t + " v585 h43z";
    case "doublevert":
      return "M145 15 v585 v" + t + ` v585 c2.667,10,9.667,15,21,15
c10,0,16.667,-5,20,-15 v-585 v` + -t + ` v-585 c-2.667,-10,-9.667,-15,-21,-15
c-10,0,-16.667,5,-20,15z M188 15 H145 v585 v` + t + ` v585 h43z
M367 15 v585 v` + t + ` v585 c2.667,10,9.667,15,21,15
c10,0,16.667,-5,20,-15 v-585 v` + -t + ` v-585 c-2.667,-10,-9.667,-15,-21,-15
c-10,0,-16.667,5,-20,15z M410 15 H367 v585 v` + t + " v585 h43z";
    case "lfloor":
      return "M319 602 V0 H403 V602 v" + t + ` v1715 h263 v84 H319z
MM319 602 V0 H403 V602 v` + t + " v1715 H319z";
    case "rfloor":
      return "M319 602 V0 H403 V602 v" + t + ` v1799 H0 v-84 H319z
MM319 602 V0 H403 V602 v` + t + " v1715 H319z";
    case "lceil":
      return "M403 1759 V84 H666 V0 H319 V1759 v" + t + ` v602 h84z
M403 1759 V0 H319 V1759 v` + t + " v602 h84z";
    case "rceil":
      return "M347 1759 V0 H0 V84 H263 V1759 v" + t + ` v602 h84z
M347 1759 V0 h-84 V1759 v` + t + " v602 h84z";
    case "lparen":
      return `M863,9c0,-2,-2,-5,-6,-9c0,0,-17,0,-17,0c-12.7,0,-19.3,0.3,-20,1
c-5.3,5.3,-10.3,11,-15,17c-242.7,294.7,-395.3,682,-458,1162c-21.3,163.3,-33.3,349,
-36,557 l0,` + (t + 84) + `c0.2,6,0,26,0,60c2,159.3,10,310.7,24,454c53.3,528,210,
949.7,470,1265c4.7,6,9.7,11.7,15,17c0.7,0.7,7,1,19,1c0,0,18,0,18,0c4,-4,6,-7,6,-9
c0,-2.7,-3.3,-8.7,-10,-18c-135.3,-192.7,-235.5,-414.3,-300.5,-665c-65,-250.7,-102.5,
-544.7,-112.5,-882c-2,-104,-3,-167,-3,-189
l0,-` + (t + 92) + `c0,-162.7,5.7,-314,17,-454c20.7,-272,63.7,-513,129,-723c65.3,
-210,155.3,-396.3,270,-559c6.7,-9.3,10,-15.3,10,-18z`;
    case "rparen":
      return `M76,0c-16.7,0,-25,3,-25,9c0,2,2,6.3,6,13c21.3,28.7,42.3,60.3,
63,95c96.7,156.7,172.8,332.5,228.5,527.5c55.7,195,92.8,416.5,111.5,664.5
c11.3,139.3,17,290.7,17,454c0,28,1.7,43,3.3,45l0,` + (t + 9) + `
c-3,4,-3.3,16.7,-3.3,38c0,162,-5.7,313.7,-17,455c-18.7,248,-55.8,469.3,-111.5,664
c-55.7,194.7,-131.8,370.3,-228.5,527c-20.7,34.7,-41.7,66.3,-63,95c-2,3.3,-4,7,-6,11
c0,7.3,5.7,11,17,11c0,0,11,0,11,0c9.3,0,14.3,-0.3,15,-1c5.3,-5.3,10.3,-11,15,-17
c242.7,-294.7,395.3,-681.7,458,-1161c21.3,-164.7,33.3,-350.7,36,-558
l0,-` + (t + 144) + `c-2,-159.3,-10,-310.7,-24,-454c-53.3,-528,-210,-949.7,
-470,-1265c-4.7,-6,-9.7,-11.7,-15,-17c-0.7,-0.7,-6.7,-1,-18,-1z`;
    default:
      throw new Error("Unknown stretchy delimiter.");
  }
};
class ur {
  // HtmlDomNode
  // Never used; needed for satisfying interface.
  constructor(e) {
    this.children = void 0, this.classes = void 0, this.height = void 0, this.depth = void 0, this.maxFontSize = void 0, this.style = void 0, this.children = e, this.classes = [], this.height = 0, this.depth = 0, this.maxFontSize = 0, this.style = {};
  }
  hasClass(e) {
    return Z.contains(this.classes, e);
  }
  /** Convert the fragment into a node. */
  toNode() {
    for (var e = document.createDocumentFragment(), t = 0; t < this.children.length; t++)
      e.appendChild(this.children[t].toNode());
    return e;
  }
  /** Convert the fragment into HTML markup. */
  toMarkup() {
    for (var e = "", t = 0; t < this.children.length; t++)
      e += this.children[t].toMarkup();
    return e;
  }
  /**
   * Converts the math node into a string, similar to innerText. Applies to
   * MathDomNode's only.
   */
  toText() {
    var e = (t) => t.toText();
    return this.children.map(e).join("");
  }
}
var zt = {
  "AMS-Regular": {
    32: [0, 0, 0, 0, 0.25],
    65: [0, 0.68889, 0, 0, 0.72222],
    66: [0, 0.68889, 0, 0, 0.66667],
    67: [0, 0.68889, 0, 0, 0.72222],
    68: [0, 0.68889, 0, 0, 0.72222],
    69: [0, 0.68889, 0, 0, 0.66667],
    70: [0, 0.68889, 0, 0, 0.61111],
    71: [0, 0.68889, 0, 0, 0.77778],
    72: [0, 0.68889, 0, 0, 0.77778],
    73: [0, 0.68889, 0, 0, 0.38889],
    74: [0.16667, 0.68889, 0, 0, 0.5],
    75: [0, 0.68889, 0, 0, 0.77778],
    76: [0, 0.68889, 0, 0, 0.66667],
    77: [0, 0.68889, 0, 0, 0.94445],
    78: [0, 0.68889, 0, 0, 0.72222],
    79: [0.16667, 0.68889, 0, 0, 0.77778],
    80: [0, 0.68889, 0, 0, 0.61111],
    81: [0.16667, 0.68889, 0, 0, 0.77778],
    82: [0, 0.68889, 0, 0, 0.72222],
    83: [0, 0.68889, 0, 0, 0.55556],
    84: [0, 0.68889, 0, 0, 0.66667],
    85: [0, 0.68889, 0, 0, 0.72222],
    86: [0, 0.68889, 0, 0, 0.72222],
    87: [0, 0.68889, 0, 0, 1],
    88: [0, 0.68889, 0, 0, 0.72222],
    89: [0, 0.68889, 0, 0, 0.72222],
    90: [0, 0.68889, 0, 0, 0.66667],
    107: [0, 0.68889, 0, 0, 0.55556],
    160: [0, 0, 0, 0, 0.25],
    165: [0, 0.675, 0.025, 0, 0.75],
    174: [0.15559, 0.69224, 0, 0, 0.94666],
    240: [0, 0.68889, 0, 0, 0.55556],
    295: [0, 0.68889, 0, 0, 0.54028],
    710: [0, 0.825, 0, 0, 2.33334],
    732: [0, 0.9, 0, 0, 2.33334],
    770: [0, 0.825, 0, 0, 2.33334],
    771: [0, 0.9, 0, 0, 2.33334],
    989: [0.08167, 0.58167, 0, 0, 0.77778],
    1008: [0, 0.43056, 0.04028, 0, 0.66667],
    8245: [0, 0.54986, 0, 0, 0.275],
    8463: [0, 0.68889, 0, 0, 0.54028],
    8487: [0, 0.68889, 0, 0, 0.72222],
    8498: [0, 0.68889, 0, 0, 0.55556],
    8502: [0, 0.68889, 0, 0, 0.66667],
    8503: [0, 0.68889, 0, 0, 0.44445],
    8504: [0, 0.68889, 0, 0, 0.66667],
    8513: [0, 0.68889, 0, 0, 0.63889],
    8592: [-0.03598, 0.46402, 0, 0, 0.5],
    8594: [-0.03598, 0.46402, 0, 0, 0.5],
    8602: [-0.13313, 0.36687, 0, 0, 1],
    8603: [-0.13313, 0.36687, 0, 0, 1],
    8606: [0.01354, 0.52239, 0, 0, 1],
    8608: [0.01354, 0.52239, 0, 0, 1],
    8610: [0.01354, 0.52239, 0, 0, 1.11111],
    8611: [0.01354, 0.52239, 0, 0, 1.11111],
    8619: [0, 0.54986, 0, 0, 1],
    8620: [0, 0.54986, 0, 0, 1],
    8621: [-0.13313, 0.37788, 0, 0, 1.38889],
    8622: [-0.13313, 0.36687, 0, 0, 1],
    8624: [0, 0.69224, 0, 0, 0.5],
    8625: [0, 0.69224, 0, 0, 0.5],
    8630: [0, 0.43056, 0, 0, 1],
    8631: [0, 0.43056, 0, 0, 1],
    8634: [0.08198, 0.58198, 0, 0, 0.77778],
    8635: [0.08198, 0.58198, 0, 0, 0.77778],
    8638: [0.19444, 0.69224, 0, 0, 0.41667],
    8639: [0.19444, 0.69224, 0, 0, 0.41667],
    8642: [0.19444, 0.69224, 0, 0, 0.41667],
    8643: [0.19444, 0.69224, 0, 0, 0.41667],
    8644: [0.1808, 0.675, 0, 0, 1],
    8646: [0.1808, 0.675, 0, 0, 1],
    8647: [0.1808, 0.675, 0, 0, 1],
    8648: [0.19444, 0.69224, 0, 0, 0.83334],
    8649: [0.1808, 0.675, 0, 0, 1],
    8650: [0.19444, 0.69224, 0, 0, 0.83334],
    8651: [0.01354, 0.52239, 0, 0, 1],
    8652: [0.01354, 0.52239, 0, 0, 1],
    8653: [-0.13313, 0.36687, 0, 0, 1],
    8654: [-0.13313, 0.36687, 0, 0, 1],
    8655: [-0.13313, 0.36687, 0, 0, 1],
    8666: [0.13667, 0.63667, 0, 0, 1],
    8667: [0.13667, 0.63667, 0, 0, 1],
    8669: [-0.13313, 0.37788, 0, 0, 1],
    8672: [-0.064, 0.437, 0, 0, 1.334],
    8674: [-0.064, 0.437, 0, 0, 1.334],
    8705: [0, 0.825, 0, 0, 0.5],
    8708: [0, 0.68889, 0, 0, 0.55556],
    8709: [0.08167, 0.58167, 0, 0, 0.77778],
    8717: [0, 0.43056, 0, 0, 0.42917],
    8722: [-0.03598, 0.46402, 0, 0, 0.5],
    8724: [0.08198, 0.69224, 0, 0, 0.77778],
    8726: [0.08167, 0.58167, 0, 0, 0.77778],
    8733: [0, 0.69224, 0, 0, 0.77778],
    8736: [0, 0.69224, 0, 0, 0.72222],
    8737: [0, 0.69224, 0, 0, 0.72222],
    8738: [0.03517, 0.52239, 0, 0, 0.72222],
    8739: [0.08167, 0.58167, 0, 0, 0.22222],
    8740: [0.25142, 0.74111, 0, 0, 0.27778],
    8741: [0.08167, 0.58167, 0, 0, 0.38889],
    8742: [0.25142, 0.74111, 0, 0, 0.5],
    8756: [0, 0.69224, 0, 0, 0.66667],
    8757: [0, 0.69224, 0, 0, 0.66667],
    8764: [-0.13313, 0.36687, 0, 0, 0.77778],
    8765: [-0.13313, 0.37788, 0, 0, 0.77778],
    8769: [-0.13313, 0.36687, 0, 0, 0.77778],
    8770: [-0.03625, 0.46375, 0, 0, 0.77778],
    8774: [0.30274, 0.79383, 0, 0, 0.77778],
    8776: [-0.01688, 0.48312, 0, 0, 0.77778],
    8778: [0.08167, 0.58167, 0, 0, 0.77778],
    8782: [0.06062, 0.54986, 0, 0, 0.77778],
    8783: [0.06062, 0.54986, 0, 0, 0.77778],
    8785: [0.08198, 0.58198, 0, 0, 0.77778],
    8786: [0.08198, 0.58198, 0, 0, 0.77778],
    8787: [0.08198, 0.58198, 0, 0, 0.77778],
    8790: [0, 0.69224, 0, 0, 0.77778],
    8791: [0.22958, 0.72958, 0, 0, 0.77778],
    8796: [0.08198, 0.91667, 0, 0, 0.77778],
    8806: [0.25583, 0.75583, 0, 0, 0.77778],
    8807: [0.25583, 0.75583, 0, 0, 0.77778],
    8808: [0.25142, 0.75726, 0, 0, 0.77778],
    8809: [0.25142, 0.75726, 0, 0, 0.77778],
    8812: [0.25583, 0.75583, 0, 0, 0.5],
    8814: [0.20576, 0.70576, 0, 0, 0.77778],
    8815: [0.20576, 0.70576, 0, 0, 0.77778],
    8816: [0.30274, 0.79383, 0, 0, 0.77778],
    8817: [0.30274, 0.79383, 0, 0, 0.77778],
    8818: [0.22958, 0.72958, 0, 0, 0.77778],
    8819: [0.22958, 0.72958, 0, 0, 0.77778],
    8822: [0.1808, 0.675, 0, 0, 0.77778],
    8823: [0.1808, 0.675, 0, 0, 0.77778],
    8828: [0.13667, 0.63667, 0, 0, 0.77778],
    8829: [0.13667, 0.63667, 0, 0, 0.77778],
    8830: [0.22958, 0.72958, 0, 0, 0.77778],
    8831: [0.22958, 0.72958, 0, 0, 0.77778],
    8832: [0.20576, 0.70576, 0, 0, 0.77778],
    8833: [0.20576, 0.70576, 0, 0, 0.77778],
    8840: [0.30274, 0.79383, 0, 0, 0.77778],
    8841: [0.30274, 0.79383, 0, 0, 0.77778],
    8842: [0.13597, 0.63597, 0, 0, 0.77778],
    8843: [0.13597, 0.63597, 0, 0, 0.77778],
    8847: [0.03517, 0.54986, 0, 0, 0.77778],
    8848: [0.03517, 0.54986, 0, 0, 0.77778],
    8858: [0.08198, 0.58198, 0, 0, 0.77778],
    8859: [0.08198, 0.58198, 0, 0, 0.77778],
    8861: [0.08198, 0.58198, 0, 0, 0.77778],
    8862: [0, 0.675, 0, 0, 0.77778],
    8863: [0, 0.675, 0, 0, 0.77778],
    8864: [0, 0.675, 0, 0, 0.77778],
    8865: [0, 0.675, 0, 0, 0.77778],
    8872: [0, 0.69224, 0, 0, 0.61111],
    8873: [0, 0.69224, 0, 0, 0.72222],
    8874: [0, 0.69224, 0, 0, 0.88889],
    8876: [0, 0.68889, 0, 0, 0.61111],
    8877: [0, 0.68889, 0, 0, 0.61111],
    8878: [0, 0.68889, 0, 0, 0.72222],
    8879: [0, 0.68889, 0, 0, 0.72222],
    8882: [0.03517, 0.54986, 0, 0, 0.77778],
    8883: [0.03517, 0.54986, 0, 0, 0.77778],
    8884: [0.13667, 0.63667, 0, 0, 0.77778],
    8885: [0.13667, 0.63667, 0, 0, 0.77778],
    8888: [0, 0.54986, 0, 0, 1.11111],
    8890: [0.19444, 0.43056, 0, 0, 0.55556],
    8891: [0.19444, 0.69224, 0, 0, 0.61111],
    8892: [0.19444, 0.69224, 0, 0, 0.61111],
    8901: [0, 0.54986, 0, 0, 0.27778],
    8903: [0.08167, 0.58167, 0, 0, 0.77778],
    8905: [0.08167, 0.58167, 0, 0, 0.77778],
    8906: [0.08167, 0.58167, 0, 0, 0.77778],
    8907: [0, 0.69224, 0, 0, 0.77778],
    8908: [0, 0.69224, 0, 0, 0.77778],
    8909: [-0.03598, 0.46402, 0, 0, 0.77778],
    8910: [0, 0.54986, 0, 0, 0.76042],
    8911: [0, 0.54986, 0, 0, 0.76042],
    8912: [0.03517, 0.54986, 0, 0, 0.77778],
    8913: [0.03517, 0.54986, 0, 0, 0.77778],
    8914: [0, 0.54986, 0, 0, 0.66667],
    8915: [0, 0.54986, 0, 0, 0.66667],
    8916: [0, 0.69224, 0, 0, 0.66667],
    8918: [0.0391, 0.5391, 0, 0, 0.77778],
    8919: [0.0391, 0.5391, 0, 0, 0.77778],
    8920: [0.03517, 0.54986, 0, 0, 1.33334],
    8921: [0.03517, 0.54986, 0, 0, 1.33334],
    8922: [0.38569, 0.88569, 0, 0, 0.77778],
    8923: [0.38569, 0.88569, 0, 0, 0.77778],
    8926: [0.13667, 0.63667, 0, 0, 0.77778],
    8927: [0.13667, 0.63667, 0, 0, 0.77778],
    8928: [0.30274, 0.79383, 0, 0, 0.77778],
    8929: [0.30274, 0.79383, 0, 0, 0.77778],
    8934: [0.23222, 0.74111, 0, 0, 0.77778],
    8935: [0.23222, 0.74111, 0, 0, 0.77778],
    8936: [0.23222, 0.74111, 0, 0, 0.77778],
    8937: [0.23222, 0.74111, 0, 0, 0.77778],
    8938: [0.20576, 0.70576, 0, 0, 0.77778],
    8939: [0.20576, 0.70576, 0, 0, 0.77778],
    8940: [0.30274, 0.79383, 0, 0, 0.77778],
    8941: [0.30274, 0.79383, 0, 0, 0.77778],
    8994: [0.19444, 0.69224, 0, 0, 0.77778],
    8995: [0.19444, 0.69224, 0, 0, 0.77778],
    9416: [0.15559, 0.69224, 0, 0, 0.90222],
    9484: [0, 0.69224, 0, 0, 0.5],
    9488: [0, 0.69224, 0, 0, 0.5],
    9492: [0, 0.37788, 0, 0, 0.5],
    9496: [0, 0.37788, 0, 0, 0.5],
    9585: [0.19444, 0.68889, 0, 0, 0.88889],
    9586: [0.19444, 0.74111, 0, 0, 0.88889],
    9632: [0, 0.675, 0, 0, 0.77778],
    9633: [0, 0.675, 0, 0, 0.77778],
    9650: [0, 0.54986, 0, 0, 0.72222],
    9651: [0, 0.54986, 0, 0, 0.72222],
    9654: [0.03517, 0.54986, 0, 0, 0.77778],
    9660: [0, 0.54986, 0, 0, 0.72222],
    9661: [0, 0.54986, 0, 0, 0.72222],
    9664: [0.03517, 0.54986, 0, 0, 0.77778],
    9674: [0.11111, 0.69224, 0, 0, 0.66667],
    9733: [0.19444, 0.69224, 0, 0, 0.94445],
    10003: [0, 0.69224, 0, 0, 0.83334],
    10016: [0, 0.69224, 0, 0, 0.83334],
    10731: [0.11111, 0.69224, 0, 0, 0.66667],
    10846: [0.19444, 0.75583, 0, 0, 0.61111],
    10877: [0.13667, 0.63667, 0, 0, 0.77778],
    10878: [0.13667, 0.63667, 0, 0, 0.77778],
    10885: [0.25583, 0.75583, 0, 0, 0.77778],
    10886: [0.25583, 0.75583, 0, 0, 0.77778],
    10887: [0.13597, 0.63597, 0, 0, 0.77778],
    10888: [0.13597, 0.63597, 0, 0, 0.77778],
    10889: [0.26167, 0.75726, 0, 0, 0.77778],
    10890: [0.26167, 0.75726, 0, 0, 0.77778],
    10891: [0.48256, 0.98256, 0, 0, 0.77778],
    10892: [0.48256, 0.98256, 0, 0, 0.77778],
    10901: [0.13667, 0.63667, 0, 0, 0.77778],
    10902: [0.13667, 0.63667, 0, 0, 0.77778],
    10933: [0.25142, 0.75726, 0, 0, 0.77778],
    10934: [0.25142, 0.75726, 0, 0, 0.77778],
    10935: [0.26167, 0.75726, 0, 0, 0.77778],
    10936: [0.26167, 0.75726, 0, 0, 0.77778],
    10937: [0.26167, 0.75726, 0, 0, 0.77778],
    10938: [0.26167, 0.75726, 0, 0, 0.77778],
    10949: [0.25583, 0.75583, 0, 0, 0.77778],
    10950: [0.25583, 0.75583, 0, 0, 0.77778],
    10955: [0.28481, 0.79383, 0, 0, 0.77778],
    10956: [0.28481, 0.79383, 0, 0, 0.77778],
    57350: [0.08167, 0.58167, 0, 0, 0.22222],
    57351: [0.08167, 0.58167, 0, 0, 0.38889],
    57352: [0.08167, 0.58167, 0, 0, 0.77778],
    57353: [0, 0.43056, 0.04028, 0, 0.66667],
    57356: [0.25142, 0.75726, 0, 0, 0.77778],
    57357: [0.25142, 0.75726, 0, 0, 0.77778],
    57358: [0.41951, 0.91951, 0, 0, 0.77778],
    57359: [0.30274, 0.79383, 0, 0, 0.77778],
    57360: [0.30274, 0.79383, 0, 0, 0.77778],
    57361: [0.41951, 0.91951, 0, 0, 0.77778],
    57366: [0.25142, 0.75726, 0, 0, 0.77778],
    57367: [0.25142, 0.75726, 0, 0, 0.77778],
    57368: [0.25142, 0.75726, 0, 0, 0.77778],
    57369: [0.25142, 0.75726, 0, 0, 0.77778],
    57370: [0.13597, 0.63597, 0, 0, 0.77778],
    57371: [0.13597, 0.63597, 0, 0, 0.77778]
  },
  "Caligraphic-Regular": {
    32: [0, 0, 0, 0, 0.25],
    65: [0, 0.68333, 0, 0.19445, 0.79847],
    66: [0, 0.68333, 0.03041, 0.13889, 0.65681],
    67: [0, 0.68333, 0.05834, 0.13889, 0.52653],
    68: [0, 0.68333, 0.02778, 0.08334, 0.77139],
    69: [0, 0.68333, 0.08944, 0.11111, 0.52778],
    70: [0, 0.68333, 0.09931, 0.11111, 0.71875],
    71: [0.09722, 0.68333, 0.0593, 0.11111, 0.59487],
    72: [0, 0.68333, 965e-5, 0.11111, 0.84452],
    73: [0, 0.68333, 0.07382, 0, 0.54452],
    74: [0.09722, 0.68333, 0.18472, 0.16667, 0.67778],
    75: [0, 0.68333, 0.01445, 0.05556, 0.76195],
    76: [0, 0.68333, 0, 0.13889, 0.68972],
    77: [0, 0.68333, 0, 0.13889, 1.2009],
    78: [0, 0.68333, 0.14736, 0.08334, 0.82049],
    79: [0, 0.68333, 0.02778, 0.11111, 0.79611],
    80: [0, 0.68333, 0.08222, 0.08334, 0.69556],
    81: [0.09722, 0.68333, 0, 0.11111, 0.81667],
    82: [0, 0.68333, 0, 0.08334, 0.8475],
    83: [0, 0.68333, 0.075, 0.13889, 0.60556],
    84: [0, 0.68333, 0.25417, 0, 0.54464],
    85: [0, 0.68333, 0.09931, 0.08334, 0.62583],
    86: [0, 0.68333, 0.08222, 0, 0.61278],
    87: [0, 0.68333, 0.08222, 0.08334, 0.98778],
    88: [0, 0.68333, 0.14643, 0.13889, 0.7133],
    89: [0.09722, 0.68333, 0.08222, 0.08334, 0.66834],
    90: [0, 0.68333, 0.07944, 0.13889, 0.72473],
    160: [0, 0, 0, 0, 0.25]
  },
  "Fraktur-Regular": {
    32: [0, 0, 0, 0, 0.25],
    33: [0, 0.69141, 0, 0, 0.29574],
    34: [0, 0.69141, 0, 0, 0.21471],
    38: [0, 0.69141, 0, 0, 0.73786],
    39: [0, 0.69141, 0, 0, 0.21201],
    40: [0.24982, 0.74947, 0, 0, 0.38865],
    41: [0.24982, 0.74947, 0, 0, 0.38865],
    42: [0, 0.62119, 0, 0, 0.27764],
    43: [0.08319, 0.58283, 0, 0, 0.75623],
    44: [0, 0.10803, 0, 0, 0.27764],
    45: [0.08319, 0.58283, 0, 0, 0.75623],
    46: [0, 0.10803, 0, 0, 0.27764],
    47: [0.24982, 0.74947, 0, 0, 0.50181],
    48: [0, 0.47534, 0, 0, 0.50181],
    49: [0, 0.47534, 0, 0, 0.50181],
    50: [0, 0.47534, 0, 0, 0.50181],
    51: [0.18906, 0.47534, 0, 0, 0.50181],
    52: [0.18906, 0.47534, 0, 0, 0.50181],
    53: [0.18906, 0.47534, 0, 0, 0.50181],
    54: [0, 0.69141, 0, 0, 0.50181],
    55: [0.18906, 0.47534, 0, 0, 0.50181],
    56: [0, 0.69141, 0, 0, 0.50181],
    57: [0.18906, 0.47534, 0, 0, 0.50181],
    58: [0, 0.47534, 0, 0, 0.21606],
    59: [0.12604, 0.47534, 0, 0, 0.21606],
    61: [-0.13099, 0.36866, 0, 0, 0.75623],
    63: [0, 0.69141, 0, 0, 0.36245],
    65: [0, 0.69141, 0, 0, 0.7176],
    66: [0, 0.69141, 0, 0, 0.88397],
    67: [0, 0.69141, 0, 0, 0.61254],
    68: [0, 0.69141, 0, 0, 0.83158],
    69: [0, 0.69141, 0, 0, 0.66278],
    70: [0.12604, 0.69141, 0, 0, 0.61119],
    71: [0, 0.69141, 0, 0, 0.78539],
    72: [0.06302, 0.69141, 0, 0, 0.7203],
    73: [0, 0.69141, 0, 0, 0.55448],
    74: [0.12604, 0.69141, 0, 0, 0.55231],
    75: [0, 0.69141, 0, 0, 0.66845],
    76: [0, 0.69141, 0, 0, 0.66602],
    77: [0, 0.69141, 0, 0, 1.04953],
    78: [0, 0.69141, 0, 0, 0.83212],
    79: [0, 0.69141, 0, 0, 0.82699],
    80: [0.18906, 0.69141, 0, 0, 0.82753],
    81: [0.03781, 0.69141, 0, 0, 0.82699],
    82: [0, 0.69141, 0, 0, 0.82807],
    83: [0, 0.69141, 0, 0, 0.82861],
    84: [0, 0.69141, 0, 0, 0.66899],
    85: [0, 0.69141, 0, 0, 0.64576],
    86: [0, 0.69141, 0, 0, 0.83131],
    87: [0, 0.69141, 0, 0, 1.04602],
    88: [0, 0.69141, 0, 0, 0.71922],
    89: [0.18906, 0.69141, 0, 0, 0.83293],
    90: [0.12604, 0.69141, 0, 0, 0.60201],
    91: [0.24982, 0.74947, 0, 0, 0.27764],
    93: [0.24982, 0.74947, 0, 0, 0.27764],
    94: [0, 0.69141, 0, 0, 0.49965],
    97: [0, 0.47534, 0, 0, 0.50046],
    98: [0, 0.69141, 0, 0, 0.51315],
    99: [0, 0.47534, 0, 0, 0.38946],
    100: [0, 0.62119, 0, 0, 0.49857],
    101: [0, 0.47534, 0, 0, 0.40053],
    102: [0.18906, 0.69141, 0, 0, 0.32626],
    103: [0.18906, 0.47534, 0, 0, 0.5037],
    104: [0.18906, 0.69141, 0, 0, 0.52126],
    105: [0, 0.69141, 0, 0, 0.27899],
    106: [0, 0.69141, 0, 0, 0.28088],
    107: [0, 0.69141, 0, 0, 0.38946],
    108: [0, 0.69141, 0, 0, 0.27953],
    109: [0, 0.47534, 0, 0, 0.76676],
    110: [0, 0.47534, 0, 0, 0.52666],
    111: [0, 0.47534, 0, 0, 0.48885],
    112: [0.18906, 0.52396, 0, 0, 0.50046],
    113: [0.18906, 0.47534, 0, 0, 0.48912],
    114: [0, 0.47534, 0, 0, 0.38919],
    115: [0, 0.47534, 0, 0, 0.44266],
    116: [0, 0.62119, 0, 0, 0.33301],
    117: [0, 0.47534, 0, 0, 0.5172],
    118: [0, 0.52396, 0, 0, 0.5118],
    119: [0, 0.52396, 0, 0, 0.77351],
    120: [0.18906, 0.47534, 0, 0, 0.38865],
    121: [0.18906, 0.47534, 0, 0, 0.49884],
    122: [0.18906, 0.47534, 0, 0, 0.39054],
    160: [0, 0, 0, 0, 0.25],
    8216: [0, 0.69141, 0, 0, 0.21471],
    8217: [0, 0.69141, 0, 0, 0.21471],
    58112: [0, 0.62119, 0, 0, 0.49749],
    58113: [0, 0.62119, 0, 0, 0.4983],
    58114: [0.18906, 0.69141, 0, 0, 0.33328],
    58115: [0.18906, 0.69141, 0, 0, 0.32923],
    58116: [0.18906, 0.47534, 0, 0, 0.50343],
    58117: [0, 0.69141, 0, 0, 0.33301],
    58118: [0, 0.62119, 0, 0, 0.33409],
    58119: [0, 0.47534, 0, 0, 0.50073]
  },
  "Main-Bold": {
    32: [0, 0, 0, 0, 0.25],
    33: [0, 0.69444, 0, 0, 0.35],
    34: [0, 0.69444, 0, 0, 0.60278],
    35: [0.19444, 0.69444, 0, 0, 0.95833],
    36: [0.05556, 0.75, 0, 0, 0.575],
    37: [0.05556, 0.75, 0, 0, 0.95833],
    38: [0, 0.69444, 0, 0, 0.89444],
    39: [0, 0.69444, 0, 0, 0.31944],
    40: [0.25, 0.75, 0, 0, 0.44722],
    41: [0.25, 0.75, 0, 0, 0.44722],
    42: [0, 0.75, 0, 0, 0.575],
    43: [0.13333, 0.63333, 0, 0, 0.89444],
    44: [0.19444, 0.15556, 0, 0, 0.31944],
    45: [0, 0.44444, 0, 0, 0.38333],
    46: [0, 0.15556, 0, 0, 0.31944],
    47: [0.25, 0.75, 0, 0, 0.575],
    48: [0, 0.64444, 0, 0, 0.575],
    49: [0, 0.64444, 0, 0, 0.575],
    50: [0, 0.64444, 0, 0, 0.575],
    51: [0, 0.64444, 0, 0, 0.575],
    52: [0, 0.64444, 0, 0, 0.575],
    53: [0, 0.64444, 0, 0, 0.575],
    54: [0, 0.64444, 0, 0, 0.575],
    55: [0, 0.64444, 0, 0, 0.575],
    56: [0, 0.64444, 0, 0, 0.575],
    57: [0, 0.64444, 0, 0, 0.575],
    58: [0, 0.44444, 0, 0, 0.31944],
    59: [0.19444, 0.44444, 0, 0, 0.31944],
    60: [0.08556, 0.58556, 0, 0, 0.89444],
    61: [-0.10889, 0.39111, 0, 0, 0.89444],
    62: [0.08556, 0.58556, 0, 0, 0.89444],
    63: [0, 0.69444, 0, 0, 0.54305],
    64: [0, 0.69444, 0, 0, 0.89444],
    65: [0, 0.68611, 0, 0, 0.86944],
    66: [0, 0.68611, 0, 0, 0.81805],
    67: [0, 0.68611, 0, 0, 0.83055],
    68: [0, 0.68611, 0, 0, 0.88194],
    69: [0, 0.68611, 0, 0, 0.75555],
    70: [0, 0.68611, 0, 0, 0.72361],
    71: [0, 0.68611, 0, 0, 0.90416],
    72: [0, 0.68611, 0, 0, 0.9],
    73: [0, 0.68611, 0, 0, 0.43611],
    74: [0, 0.68611, 0, 0, 0.59444],
    75: [0, 0.68611, 0, 0, 0.90138],
    76: [0, 0.68611, 0, 0, 0.69166],
    77: [0, 0.68611, 0, 0, 1.09166],
    78: [0, 0.68611, 0, 0, 0.9],
    79: [0, 0.68611, 0, 0, 0.86388],
    80: [0, 0.68611, 0, 0, 0.78611],
    81: [0.19444, 0.68611, 0, 0, 0.86388],
    82: [0, 0.68611, 0, 0, 0.8625],
    83: [0, 0.68611, 0, 0, 0.63889],
    84: [0, 0.68611, 0, 0, 0.8],
    85: [0, 0.68611, 0, 0, 0.88472],
    86: [0, 0.68611, 0.01597, 0, 0.86944],
    87: [0, 0.68611, 0.01597, 0, 1.18888],
    88: [0, 0.68611, 0, 0, 0.86944],
    89: [0, 0.68611, 0.02875, 0, 0.86944],
    90: [0, 0.68611, 0, 0, 0.70277],
    91: [0.25, 0.75, 0, 0, 0.31944],
    92: [0.25, 0.75, 0, 0, 0.575],
    93: [0.25, 0.75, 0, 0, 0.31944],
    94: [0, 0.69444, 0, 0, 0.575],
    95: [0.31, 0.13444, 0.03194, 0, 0.575],
    97: [0, 0.44444, 0, 0, 0.55902],
    98: [0, 0.69444, 0, 0, 0.63889],
    99: [0, 0.44444, 0, 0, 0.51111],
    100: [0, 0.69444, 0, 0, 0.63889],
    101: [0, 0.44444, 0, 0, 0.52708],
    102: [0, 0.69444, 0.10903, 0, 0.35139],
    103: [0.19444, 0.44444, 0.01597, 0, 0.575],
    104: [0, 0.69444, 0, 0, 0.63889],
    105: [0, 0.69444, 0, 0, 0.31944],
    106: [0.19444, 0.69444, 0, 0, 0.35139],
    107: [0, 0.69444, 0, 0, 0.60694],
    108: [0, 0.69444, 0, 0, 0.31944],
    109: [0, 0.44444, 0, 0, 0.95833],
    110: [0, 0.44444, 0, 0, 0.63889],
    111: [0, 0.44444, 0, 0, 0.575],
    112: [0.19444, 0.44444, 0, 0, 0.63889],
    113: [0.19444, 0.44444, 0, 0, 0.60694],
    114: [0, 0.44444, 0, 0, 0.47361],
    115: [0, 0.44444, 0, 0, 0.45361],
    116: [0, 0.63492, 0, 0, 0.44722],
    117: [0, 0.44444, 0, 0, 0.63889],
    118: [0, 0.44444, 0.01597, 0, 0.60694],
    119: [0, 0.44444, 0.01597, 0, 0.83055],
    120: [0, 0.44444, 0, 0, 0.60694],
    121: [0.19444, 0.44444, 0.01597, 0, 0.60694],
    122: [0, 0.44444, 0, 0, 0.51111],
    123: [0.25, 0.75, 0, 0, 0.575],
    124: [0.25, 0.75, 0, 0, 0.31944],
    125: [0.25, 0.75, 0, 0, 0.575],
    126: [0.35, 0.34444, 0, 0, 0.575],
    160: [0, 0, 0, 0, 0.25],
    163: [0, 0.69444, 0, 0, 0.86853],
    168: [0, 0.69444, 0, 0, 0.575],
    172: [0, 0.44444, 0, 0, 0.76666],
    176: [0, 0.69444, 0, 0, 0.86944],
    177: [0.13333, 0.63333, 0, 0, 0.89444],
    184: [0.17014, 0, 0, 0, 0.51111],
    198: [0, 0.68611, 0, 0, 1.04166],
    215: [0.13333, 0.63333, 0, 0, 0.89444],
    216: [0.04861, 0.73472, 0, 0, 0.89444],
    223: [0, 0.69444, 0, 0, 0.59722],
    230: [0, 0.44444, 0, 0, 0.83055],
    247: [0.13333, 0.63333, 0, 0, 0.89444],
    248: [0.09722, 0.54167, 0, 0, 0.575],
    305: [0, 0.44444, 0, 0, 0.31944],
    338: [0, 0.68611, 0, 0, 1.16944],
    339: [0, 0.44444, 0, 0, 0.89444],
    567: [0.19444, 0.44444, 0, 0, 0.35139],
    710: [0, 0.69444, 0, 0, 0.575],
    711: [0, 0.63194, 0, 0, 0.575],
    713: [0, 0.59611, 0, 0, 0.575],
    714: [0, 0.69444, 0, 0, 0.575],
    715: [0, 0.69444, 0, 0, 0.575],
    728: [0, 0.69444, 0, 0, 0.575],
    729: [0, 0.69444, 0, 0, 0.31944],
    730: [0, 0.69444, 0, 0, 0.86944],
    732: [0, 0.69444, 0, 0, 0.575],
    733: [0, 0.69444, 0, 0, 0.575],
    915: [0, 0.68611, 0, 0, 0.69166],
    916: [0, 0.68611, 0, 0, 0.95833],
    920: [0, 0.68611, 0, 0, 0.89444],
    923: [0, 0.68611, 0, 0, 0.80555],
    926: [0, 0.68611, 0, 0, 0.76666],
    928: [0, 0.68611, 0, 0, 0.9],
    931: [0, 0.68611, 0, 0, 0.83055],
    933: [0, 0.68611, 0, 0, 0.89444],
    934: [0, 0.68611, 0, 0, 0.83055],
    936: [0, 0.68611, 0, 0, 0.89444],
    937: [0, 0.68611, 0, 0, 0.83055],
    8211: [0, 0.44444, 0.03194, 0, 0.575],
    8212: [0, 0.44444, 0.03194, 0, 1.14999],
    8216: [0, 0.69444, 0, 0, 0.31944],
    8217: [0, 0.69444, 0, 0, 0.31944],
    8220: [0, 0.69444, 0, 0, 0.60278],
    8221: [0, 0.69444, 0, 0, 0.60278],
    8224: [0.19444, 0.69444, 0, 0, 0.51111],
    8225: [0.19444, 0.69444, 0, 0, 0.51111],
    8242: [0, 0.55556, 0, 0, 0.34444],
    8407: [0, 0.72444, 0.15486, 0, 0.575],
    8463: [0, 0.69444, 0, 0, 0.66759],
    8465: [0, 0.69444, 0, 0, 0.83055],
    8467: [0, 0.69444, 0, 0, 0.47361],
    8472: [0.19444, 0.44444, 0, 0, 0.74027],
    8476: [0, 0.69444, 0, 0, 0.83055],
    8501: [0, 0.69444, 0, 0, 0.70277],
    8592: [-0.10889, 0.39111, 0, 0, 1.14999],
    8593: [0.19444, 0.69444, 0, 0, 0.575],
    8594: [-0.10889, 0.39111, 0, 0, 1.14999],
    8595: [0.19444, 0.69444, 0, 0, 0.575],
    8596: [-0.10889, 0.39111, 0, 0, 1.14999],
    8597: [0.25, 0.75, 0, 0, 0.575],
    8598: [0.19444, 0.69444, 0, 0, 1.14999],
    8599: [0.19444, 0.69444, 0, 0, 1.14999],
    8600: [0.19444, 0.69444, 0, 0, 1.14999],
    8601: [0.19444, 0.69444, 0, 0, 1.14999],
    8636: [-0.10889, 0.39111, 0, 0, 1.14999],
    8637: [-0.10889, 0.39111, 0, 0, 1.14999],
    8640: [-0.10889, 0.39111, 0, 0, 1.14999],
    8641: [-0.10889, 0.39111, 0, 0, 1.14999],
    8656: [-0.10889, 0.39111, 0, 0, 1.14999],
    8657: [0.19444, 0.69444, 0, 0, 0.70277],
    8658: [-0.10889, 0.39111, 0, 0, 1.14999],
    8659: [0.19444, 0.69444, 0, 0, 0.70277],
    8660: [-0.10889, 0.39111, 0, 0, 1.14999],
    8661: [0.25, 0.75, 0, 0, 0.70277],
    8704: [0, 0.69444, 0, 0, 0.63889],
    8706: [0, 0.69444, 0.06389, 0, 0.62847],
    8707: [0, 0.69444, 0, 0, 0.63889],
    8709: [0.05556, 0.75, 0, 0, 0.575],
    8711: [0, 0.68611, 0, 0, 0.95833],
    8712: [0.08556, 0.58556, 0, 0, 0.76666],
    8715: [0.08556, 0.58556, 0, 0, 0.76666],
    8722: [0.13333, 0.63333, 0, 0, 0.89444],
    8723: [0.13333, 0.63333, 0, 0, 0.89444],
    8725: [0.25, 0.75, 0, 0, 0.575],
    8726: [0.25, 0.75, 0, 0, 0.575],
    8727: [-0.02778, 0.47222, 0, 0, 0.575],
    8728: [-0.02639, 0.47361, 0, 0, 0.575],
    8729: [-0.02639, 0.47361, 0, 0, 0.575],
    8730: [0.18, 0.82, 0, 0, 0.95833],
    8733: [0, 0.44444, 0, 0, 0.89444],
    8734: [0, 0.44444, 0, 0, 1.14999],
    8736: [0, 0.69224, 0, 0, 0.72222],
    8739: [0.25, 0.75, 0, 0, 0.31944],
    8741: [0.25, 0.75, 0, 0, 0.575],
    8743: [0, 0.55556, 0, 0, 0.76666],
    8744: [0, 0.55556, 0, 0, 0.76666],
    8745: [0, 0.55556, 0, 0, 0.76666],
    8746: [0, 0.55556, 0, 0, 0.76666],
    8747: [0.19444, 0.69444, 0.12778, 0, 0.56875],
    8764: [-0.10889, 0.39111, 0, 0, 0.89444],
    8768: [0.19444, 0.69444, 0, 0, 0.31944],
    8771: [222e-5, 0.50222, 0, 0, 0.89444],
    8773: [0.027, 0.638, 0, 0, 0.894],
    8776: [0.02444, 0.52444, 0, 0, 0.89444],
    8781: [222e-5, 0.50222, 0, 0, 0.89444],
    8801: [222e-5, 0.50222, 0, 0, 0.89444],
    8804: [0.19667, 0.69667, 0, 0, 0.89444],
    8805: [0.19667, 0.69667, 0, 0, 0.89444],
    8810: [0.08556, 0.58556, 0, 0, 1.14999],
    8811: [0.08556, 0.58556, 0, 0, 1.14999],
    8826: [0.08556, 0.58556, 0, 0, 0.89444],
    8827: [0.08556, 0.58556, 0, 0, 0.89444],
    8834: [0.08556, 0.58556, 0, 0, 0.89444],
    8835: [0.08556, 0.58556, 0, 0, 0.89444],
    8838: [0.19667, 0.69667, 0, 0, 0.89444],
    8839: [0.19667, 0.69667, 0, 0, 0.89444],
    8846: [0, 0.55556, 0, 0, 0.76666],
    8849: [0.19667, 0.69667, 0, 0, 0.89444],
    8850: [0.19667, 0.69667, 0, 0, 0.89444],
    8851: [0, 0.55556, 0, 0, 0.76666],
    8852: [0, 0.55556, 0, 0, 0.76666],
    8853: [0.13333, 0.63333, 0, 0, 0.89444],
    8854: [0.13333, 0.63333, 0, 0, 0.89444],
    8855: [0.13333, 0.63333, 0, 0, 0.89444],
    8856: [0.13333, 0.63333, 0, 0, 0.89444],
    8857: [0.13333, 0.63333, 0, 0, 0.89444],
    8866: [0, 0.69444, 0, 0, 0.70277],
    8867: [0, 0.69444, 0, 0, 0.70277],
    8868: [0, 0.69444, 0, 0, 0.89444],
    8869: [0, 0.69444, 0, 0, 0.89444],
    8900: [-0.02639, 0.47361, 0, 0, 0.575],
    8901: [-0.02639, 0.47361, 0, 0, 0.31944],
    8902: [-0.02778, 0.47222, 0, 0, 0.575],
    8968: [0.25, 0.75, 0, 0, 0.51111],
    8969: [0.25, 0.75, 0, 0, 0.51111],
    8970: [0.25, 0.75, 0, 0, 0.51111],
    8971: [0.25, 0.75, 0, 0, 0.51111],
    8994: [-0.13889, 0.36111, 0, 0, 1.14999],
    8995: [-0.13889, 0.36111, 0, 0, 1.14999],
    9651: [0.19444, 0.69444, 0, 0, 1.02222],
    9657: [-0.02778, 0.47222, 0, 0, 0.575],
    9661: [0.19444, 0.69444, 0, 0, 1.02222],
    9667: [-0.02778, 0.47222, 0, 0, 0.575],
    9711: [0.19444, 0.69444, 0, 0, 1.14999],
    9824: [0.12963, 0.69444, 0, 0, 0.89444],
    9825: [0.12963, 0.69444, 0, 0, 0.89444],
    9826: [0.12963, 0.69444, 0, 0, 0.89444],
    9827: [0.12963, 0.69444, 0, 0, 0.89444],
    9837: [0, 0.75, 0, 0, 0.44722],
    9838: [0.19444, 0.69444, 0, 0, 0.44722],
    9839: [0.19444, 0.69444, 0, 0, 0.44722],
    10216: [0.25, 0.75, 0, 0, 0.44722],
    10217: [0.25, 0.75, 0, 0, 0.44722],
    10815: [0, 0.68611, 0, 0, 0.9],
    10927: [0.19667, 0.69667, 0, 0, 0.89444],
    10928: [0.19667, 0.69667, 0, 0, 0.89444],
    57376: [0.19444, 0.69444, 0, 0, 0]
  },
  "Main-BoldItalic": {
    32: [0, 0, 0, 0, 0.25],
    33: [0, 0.69444, 0.11417, 0, 0.38611],
    34: [0, 0.69444, 0.07939, 0, 0.62055],
    35: [0.19444, 0.69444, 0.06833, 0, 0.94444],
    37: [0.05556, 0.75, 0.12861, 0, 0.94444],
    38: [0, 0.69444, 0.08528, 0, 0.88555],
    39: [0, 0.69444, 0.12945, 0, 0.35555],
    40: [0.25, 0.75, 0.15806, 0, 0.47333],
    41: [0.25, 0.75, 0.03306, 0, 0.47333],
    42: [0, 0.75, 0.14333, 0, 0.59111],
    43: [0.10333, 0.60333, 0.03306, 0, 0.88555],
    44: [0.19444, 0.14722, 0, 0, 0.35555],
    45: [0, 0.44444, 0.02611, 0, 0.41444],
    46: [0, 0.14722, 0, 0, 0.35555],
    47: [0.25, 0.75, 0.15806, 0, 0.59111],
    48: [0, 0.64444, 0.13167, 0, 0.59111],
    49: [0, 0.64444, 0.13167, 0, 0.59111],
    50: [0, 0.64444, 0.13167, 0, 0.59111],
    51: [0, 0.64444, 0.13167, 0, 0.59111],
    52: [0.19444, 0.64444, 0.13167, 0, 0.59111],
    53: [0, 0.64444, 0.13167, 0, 0.59111],
    54: [0, 0.64444, 0.13167, 0, 0.59111],
    55: [0.19444, 0.64444, 0.13167, 0, 0.59111],
    56: [0, 0.64444, 0.13167, 0, 0.59111],
    57: [0, 0.64444, 0.13167, 0, 0.59111],
    58: [0, 0.44444, 0.06695, 0, 0.35555],
    59: [0.19444, 0.44444, 0.06695, 0, 0.35555],
    61: [-0.10889, 0.39111, 0.06833, 0, 0.88555],
    63: [0, 0.69444, 0.11472, 0, 0.59111],
    64: [0, 0.69444, 0.09208, 0, 0.88555],
    65: [0, 0.68611, 0, 0, 0.86555],
    66: [0, 0.68611, 0.0992, 0, 0.81666],
    67: [0, 0.68611, 0.14208, 0, 0.82666],
    68: [0, 0.68611, 0.09062, 0, 0.87555],
    69: [0, 0.68611, 0.11431, 0, 0.75666],
    70: [0, 0.68611, 0.12903, 0, 0.72722],
    71: [0, 0.68611, 0.07347, 0, 0.89527],
    72: [0, 0.68611, 0.17208, 0, 0.8961],
    73: [0, 0.68611, 0.15681, 0, 0.47166],
    74: [0, 0.68611, 0.145, 0, 0.61055],
    75: [0, 0.68611, 0.14208, 0, 0.89499],
    76: [0, 0.68611, 0, 0, 0.69777],
    77: [0, 0.68611, 0.17208, 0, 1.07277],
    78: [0, 0.68611, 0.17208, 0, 0.8961],
    79: [0, 0.68611, 0.09062, 0, 0.85499],
    80: [0, 0.68611, 0.0992, 0, 0.78721],
    81: [0.19444, 0.68611, 0.09062, 0, 0.85499],
    82: [0, 0.68611, 0.02559, 0, 0.85944],
    83: [0, 0.68611, 0.11264, 0, 0.64999],
    84: [0, 0.68611, 0.12903, 0, 0.7961],
    85: [0, 0.68611, 0.17208, 0, 0.88083],
    86: [0, 0.68611, 0.18625, 0, 0.86555],
    87: [0, 0.68611, 0.18625, 0, 1.15999],
    88: [0, 0.68611, 0.15681, 0, 0.86555],
    89: [0, 0.68611, 0.19803, 0, 0.86555],
    90: [0, 0.68611, 0.14208, 0, 0.70888],
    91: [0.25, 0.75, 0.1875, 0, 0.35611],
    93: [0.25, 0.75, 0.09972, 0, 0.35611],
    94: [0, 0.69444, 0.06709, 0, 0.59111],
    95: [0.31, 0.13444, 0.09811, 0, 0.59111],
    97: [0, 0.44444, 0.09426, 0, 0.59111],
    98: [0, 0.69444, 0.07861, 0, 0.53222],
    99: [0, 0.44444, 0.05222, 0, 0.53222],
    100: [0, 0.69444, 0.10861, 0, 0.59111],
    101: [0, 0.44444, 0.085, 0, 0.53222],
    102: [0.19444, 0.69444, 0.21778, 0, 0.4],
    103: [0.19444, 0.44444, 0.105, 0, 0.53222],
    104: [0, 0.69444, 0.09426, 0, 0.59111],
    105: [0, 0.69326, 0.11387, 0, 0.35555],
    106: [0.19444, 0.69326, 0.1672, 0, 0.35555],
    107: [0, 0.69444, 0.11111, 0, 0.53222],
    108: [0, 0.69444, 0.10861, 0, 0.29666],
    109: [0, 0.44444, 0.09426, 0, 0.94444],
    110: [0, 0.44444, 0.09426, 0, 0.64999],
    111: [0, 0.44444, 0.07861, 0, 0.59111],
    112: [0.19444, 0.44444, 0.07861, 0, 0.59111],
    113: [0.19444, 0.44444, 0.105, 0, 0.53222],
    114: [0, 0.44444, 0.11111, 0, 0.50167],
    115: [0, 0.44444, 0.08167, 0, 0.48694],
    116: [0, 0.63492, 0.09639, 0, 0.385],
    117: [0, 0.44444, 0.09426, 0, 0.62055],
    118: [0, 0.44444, 0.11111, 0, 0.53222],
    119: [0, 0.44444, 0.11111, 0, 0.76777],
    120: [0, 0.44444, 0.12583, 0, 0.56055],
    121: [0.19444, 0.44444, 0.105, 0, 0.56166],
    122: [0, 0.44444, 0.13889, 0, 0.49055],
    126: [0.35, 0.34444, 0.11472, 0, 0.59111],
    160: [0, 0, 0, 0, 0.25],
    168: [0, 0.69444, 0.11473, 0, 0.59111],
    176: [0, 0.69444, 0, 0, 0.94888],
    184: [0.17014, 0, 0, 0, 0.53222],
    198: [0, 0.68611, 0.11431, 0, 1.02277],
    216: [0.04861, 0.73472, 0.09062, 0, 0.88555],
    223: [0.19444, 0.69444, 0.09736, 0, 0.665],
    230: [0, 0.44444, 0.085, 0, 0.82666],
    248: [0.09722, 0.54167, 0.09458, 0, 0.59111],
    305: [0, 0.44444, 0.09426, 0, 0.35555],
    338: [0, 0.68611, 0.11431, 0, 1.14054],
    339: [0, 0.44444, 0.085, 0, 0.82666],
    567: [0.19444, 0.44444, 0.04611, 0, 0.385],
    710: [0, 0.69444, 0.06709, 0, 0.59111],
    711: [0, 0.63194, 0.08271, 0, 0.59111],
    713: [0, 0.59444, 0.10444, 0, 0.59111],
    714: [0, 0.69444, 0.08528, 0, 0.59111],
    715: [0, 0.69444, 0, 0, 0.59111],
    728: [0, 0.69444, 0.10333, 0, 0.59111],
    729: [0, 0.69444, 0.12945, 0, 0.35555],
    730: [0, 0.69444, 0, 0, 0.94888],
    732: [0, 0.69444, 0.11472, 0, 0.59111],
    733: [0, 0.69444, 0.11472, 0, 0.59111],
    915: [0, 0.68611, 0.12903, 0, 0.69777],
    916: [0, 0.68611, 0, 0, 0.94444],
    920: [0, 0.68611, 0.09062, 0, 0.88555],
    923: [0, 0.68611, 0, 0, 0.80666],
    926: [0, 0.68611, 0.15092, 0, 0.76777],
    928: [0, 0.68611, 0.17208, 0, 0.8961],
    931: [0, 0.68611, 0.11431, 0, 0.82666],
    933: [0, 0.68611, 0.10778, 0, 0.88555],
    934: [0, 0.68611, 0.05632, 0, 0.82666],
    936: [0, 0.68611, 0.10778, 0, 0.88555],
    937: [0, 0.68611, 0.0992, 0, 0.82666],
    8211: [0, 0.44444, 0.09811, 0, 0.59111],
    8212: [0, 0.44444, 0.09811, 0, 1.18221],
    8216: [0, 0.69444, 0.12945, 0, 0.35555],
    8217: [0, 0.69444, 0.12945, 0, 0.35555],
    8220: [0, 0.69444, 0.16772, 0, 0.62055],
    8221: [0, 0.69444, 0.07939, 0, 0.62055]
  },
  "Main-Italic": {
    32: [0, 0, 0, 0, 0.25],
    33: [0, 0.69444, 0.12417, 0, 0.30667],
    34: [0, 0.69444, 0.06961, 0, 0.51444],
    35: [0.19444, 0.69444, 0.06616, 0, 0.81777],
    37: [0.05556, 0.75, 0.13639, 0, 0.81777],
    38: [0, 0.69444, 0.09694, 0, 0.76666],
    39: [0, 0.69444, 0.12417, 0, 0.30667],
    40: [0.25, 0.75, 0.16194, 0, 0.40889],
    41: [0.25, 0.75, 0.03694, 0, 0.40889],
    42: [0, 0.75, 0.14917, 0, 0.51111],
    43: [0.05667, 0.56167, 0.03694, 0, 0.76666],
    44: [0.19444, 0.10556, 0, 0, 0.30667],
    45: [0, 0.43056, 0.02826, 0, 0.35778],
    46: [0, 0.10556, 0, 0, 0.30667],
    47: [0.25, 0.75, 0.16194, 0, 0.51111],
    48: [0, 0.64444, 0.13556, 0, 0.51111],
    49: [0, 0.64444, 0.13556, 0, 0.51111],
    50: [0, 0.64444, 0.13556, 0, 0.51111],
    51: [0, 0.64444, 0.13556, 0, 0.51111],
    52: [0.19444, 0.64444, 0.13556, 0, 0.51111],
    53: [0, 0.64444, 0.13556, 0, 0.51111],
    54: [0, 0.64444, 0.13556, 0, 0.51111],
    55: [0.19444, 0.64444, 0.13556, 0, 0.51111],
    56: [0, 0.64444, 0.13556, 0, 0.51111],
    57: [0, 0.64444, 0.13556, 0, 0.51111],
    58: [0, 0.43056, 0.0582, 0, 0.30667],
    59: [0.19444, 0.43056, 0.0582, 0, 0.30667],
    61: [-0.13313, 0.36687, 0.06616, 0, 0.76666],
    63: [0, 0.69444, 0.1225, 0, 0.51111],
    64: [0, 0.69444, 0.09597, 0, 0.76666],
    65: [0, 0.68333, 0, 0, 0.74333],
    66: [0, 0.68333, 0.10257, 0, 0.70389],
    67: [0, 0.68333, 0.14528, 0, 0.71555],
    68: [0, 0.68333, 0.09403, 0, 0.755],
    69: [0, 0.68333, 0.12028, 0, 0.67833],
    70: [0, 0.68333, 0.13305, 0, 0.65277],
    71: [0, 0.68333, 0.08722, 0, 0.77361],
    72: [0, 0.68333, 0.16389, 0, 0.74333],
    73: [0, 0.68333, 0.15806, 0, 0.38555],
    74: [0, 0.68333, 0.14028, 0, 0.525],
    75: [0, 0.68333, 0.14528, 0, 0.76888],
    76: [0, 0.68333, 0, 0, 0.62722],
    77: [0, 0.68333, 0.16389, 0, 0.89666],
    78: [0, 0.68333, 0.16389, 0, 0.74333],
    79: [0, 0.68333, 0.09403, 0, 0.76666],
    80: [0, 0.68333, 0.10257, 0, 0.67833],
    81: [0.19444, 0.68333, 0.09403, 0, 0.76666],
    82: [0, 0.68333, 0.03868, 0, 0.72944],
    83: [0, 0.68333, 0.11972, 0, 0.56222],
    84: [0, 0.68333, 0.13305, 0, 0.71555],
    85: [0, 0.68333, 0.16389, 0, 0.74333],
    86: [0, 0.68333, 0.18361, 0, 0.74333],
    87: [0, 0.68333, 0.18361, 0, 0.99888],
    88: [0, 0.68333, 0.15806, 0, 0.74333],
    89: [0, 0.68333, 0.19383, 0, 0.74333],
    90: [0, 0.68333, 0.14528, 0, 0.61333],
    91: [0.25, 0.75, 0.1875, 0, 0.30667],
    93: [0.25, 0.75, 0.10528, 0, 0.30667],
    94: [0, 0.69444, 0.06646, 0, 0.51111],
    95: [0.31, 0.12056, 0.09208, 0, 0.51111],
    97: [0, 0.43056, 0.07671, 0, 0.51111],
    98: [0, 0.69444, 0.06312, 0, 0.46],
    99: [0, 0.43056, 0.05653, 0, 0.46],
    100: [0, 0.69444, 0.10333, 0, 0.51111],
    101: [0, 0.43056, 0.07514, 0, 0.46],
    102: [0.19444, 0.69444, 0.21194, 0, 0.30667],
    103: [0.19444, 0.43056, 0.08847, 0, 0.46],
    104: [0, 0.69444, 0.07671, 0, 0.51111],
    105: [0, 0.65536, 0.1019, 0, 0.30667],
    106: [0.19444, 0.65536, 0.14467, 0, 0.30667],
    107: [0, 0.69444, 0.10764, 0, 0.46],
    108: [0, 0.69444, 0.10333, 0, 0.25555],
    109: [0, 0.43056, 0.07671, 0, 0.81777],
    110: [0, 0.43056, 0.07671, 0, 0.56222],
    111: [0, 0.43056, 0.06312, 0, 0.51111],
    112: [0.19444, 0.43056, 0.06312, 0, 0.51111],
    113: [0.19444, 0.43056, 0.08847, 0, 0.46],
    114: [0, 0.43056, 0.10764, 0, 0.42166],
    115: [0, 0.43056, 0.08208, 0, 0.40889],
    116: [0, 0.61508, 0.09486, 0, 0.33222],
    117: [0, 0.43056, 0.07671, 0, 0.53666],
    118: [0, 0.43056, 0.10764, 0, 0.46],
    119: [0, 0.43056, 0.10764, 0, 0.66444],
    120: [0, 0.43056, 0.12042, 0, 0.46389],
    121: [0.19444, 0.43056, 0.08847, 0, 0.48555],
    122: [0, 0.43056, 0.12292, 0, 0.40889],
    126: [0.35, 0.31786, 0.11585, 0, 0.51111],
    160: [0, 0, 0, 0, 0.25],
    168: [0, 0.66786, 0.10474, 0, 0.51111],
    176: [0, 0.69444, 0, 0, 0.83129],
    184: [0.17014, 0, 0, 0, 0.46],
    198: [0, 0.68333, 0.12028, 0, 0.88277],
    216: [0.04861, 0.73194, 0.09403, 0, 0.76666],
    223: [0.19444, 0.69444, 0.10514, 0, 0.53666],
    230: [0, 0.43056, 0.07514, 0, 0.71555],
    248: [0.09722, 0.52778, 0.09194, 0, 0.51111],
    338: [0, 0.68333, 0.12028, 0, 0.98499],
    339: [0, 0.43056, 0.07514, 0, 0.71555],
    710: [0, 0.69444, 0.06646, 0, 0.51111],
    711: [0, 0.62847, 0.08295, 0, 0.51111],
    713: [0, 0.56167, 0.10333, 0, 0.51111],
    714: [0, 0.69444, 0.09694, 0, 0.51111],
    715: [0, 0.69444, 0, 0, 0.51111],
    728: [0, 0.69444, 0.10806, 0, 0.51111],
    729: [0, 0.66786, 0.11752, 0, 0.30667],
    730: [0, 0.69444, 0, 0, 0.83129],
    732: [0, 0.66786, 0.11585, 0, 0.51111],
    733: [0, 0.69444, 0.1225, 0, 0.51111],
    915: [0, 0.68333, 0.13305, 0, 0.62722],
    916: [0, 0.68333, 0, 0, 0.81777],
    920: [0, 0.68333, 0.09403, 0, 0.76666],
    923: [0, 0.68333, 0, 0, 0.69222],
    926: [0, 0.68333, 0.15294, 0, 0.66444],
    928: [0, 0.68333, 0.16389, 0, 0.74333],
    931: [0, 0.68333, 0.12028, 0, 0.71555],
    933: [0, 0.68333, 0.11111, 0, 0.76666],
    934: [0, 0.68333, 0.05986, 0, 0.71555],
    936: [0, 0.68333, 0.11111, 0, 0.76666],
    937: [0, 0.68333, 0.10257, 0, 0.71555],
    8211: [0, 0.43056, 0.09208, 0, 0.51111],
    8212: [0, 0.43056, 0.09208, 0, 1.02222],
    8216: [0, 0.69444, 0.12417, 0, 0.30667],
    8217: [0, 0.69444, 0.12417, 0, 0.30667],
    8220: [0, 0.69444, 0.1685, 0, 0.51444],
    8221: [0, 0.69444, 0.06961, 0, 0.51444],
    8463: [0, 0.68889, 0, 0, 0.54028]
  },
  "Main-Regular": {
    32: [0, 0, 0, 0, 0.25],
    33: [0, 0.69444, 0, 0, 0.27778],
    34: [0, 0.69444, 0, 0, 0.5],
    35: [0.19444, 0.69444, 0, 0, 0.83334],
    36: [0.05556, 0.75, 0, 0, 0.5],
    37: [0.05556, 0.75, 0, 0, 0.83334],
    38: [0, 0.69444, 0, 0, 0.77778],
    39: [0, 0.69444, 0, 0, 0.27778],
    40: [0.25, 0.75, 0, 0, 0.38889],
    41: [0.25, 0.75, 0, 0, 0.38889],
    42: [0, 0.75, 0, 0, 0.5],
    43: [0.08333, 0.58333, 0, 0, 0.77778],
    44: [0.19444, 0.10556, 0, 0, 0.27778],
    45: [0, 0.43056, 0, 0, 0.33333],
    46: [0, 0.10556, 0, 0, 0.27778],
    47: [0.25, 0.75, 0, 0, 0.5],
    48: [0, 0.64444, 0, 0, 0.5],
    49: [0, 0.64444, 0, 0, 0.5],
    50: [0, 0.64444, 0, 0, 0.5],
    51: [0, 0.64444, 0, 0, 0.5],
    52: [0, 0.64444, 0, 0, 0.5],
    53: [0, 0.64444, 0, 0, 0.5],
    54: [0, 0.64444, 0, 0, 0.5],
    55: [0, 0.64444, 0, 0, 0.5],
    56: [0, 0.64444, 0, 0, 0.5],
    57: [0, 0.64444, 0, 0, 0.5],
    58: [0, 0.43056, 0, 0, 0.27778],
    59: [0.19444, 0.43056, 0, 0, 0.27778],
    60: [0.0391, 0.5391, 0, 0, 0.77778],
    61: [-0.13313, 0.36687, 0, 0, 0.77778],
    62: [0.0391, 0.5391, 0, 0, 0.77778],
    63: [0, 0.69444, 0, 0, 0.47222],
    64: [0, 0.69444, 0, 0, 0.77778],
    65: [0, 0.68333, 0, 0, 0.75],
    66: [0, 0.68333, 0, 0, 0.70834],
    67: [0, 0.68333, 0, 0, 0.72222],
    68: [0, 0.68333, 0, 0, 0.76389],
    69: [0, 0.68333, 0, 0, 0.68056],
    70: [0, 0.68333, 0, 0, 0.65278],
    71: [0, 0.68333, 0, 0, 0.78472],
    72: [0, 0.68333, 0, 0, 0.75],
    73: [0, 0.68333, 0, 0, 0.36111],
    74: [0, 0.68333, 0, 0, 0.51389],
    75: [0, 0.68333, 0, 0, 0.77778],
    76: [0, 0.68333, 0, 0, 0.625],
    77: [0, 0.68333, 0, 0, 0.91667],
    78: [0, 0.68333, 0, 0, 0.75],
    79: [0, 0.68333, 0, 0, 0.77778],
    80: [0, 0.68333, 0, 0, 0.68056],
    81: [0.19444, 0.68333, 0, 0, 0.77778],
    82: [0, 0.68333, 0, 0, 0.73611],
    83: [0, 0.68333, 0, 0, 0.55556],
    84: [0, 0.68333, 0, 0, 0.72222],
    85: [0, 0.68333, 0, 0, 0.75],
    86: [0, 0.68333, 0.01389, 0, 0.75],
    87: [0, 0.68333, 0.01389, 0, 1.02778],
    88: [0, 0.68333, 0, 0, 0.75],
    89: [0, 0.68333, 0.025, 0, 0.75],
    90: [0, 0.68333, 0, 0, 0.61111],
    91: [0.25, 0.75, 0, 0, 0.27778],
    92: [0.25, 0.75, 0, 0, 0.5],
    93: [0.25, 0.75, 0, 0, 0.27778],
    94: [0, 0.69444, 0, 0, 0.5],
    95: [0.31, 0.12056, 0.02778, 0, 0.5],
    97: [0, 0.43056, 0, 0, 0.5],
    98: [0, 0.69444, 0, 0, 0.55556],
    99: [0, 0.43056, 0, 0, 0.44445],
    100: [0, 0.69444, 0, 0, 0.55556],
    101: [0, 0.43056, 0, 0, 0.44445],
    102: [0, 0.69444, 0.07778, 0, 0.30556],
    103: [0.19444, 0.43056, 0.01389, 0, 0.5],
    104: [0, 0.69444, 0, 0, 0.55556],
    105: [0, 0.66786, 0, 0, 0.27778],
    106: [0.19444, 0.66786, 0, 0, 0.30556],
    107: [0, 0.69444, 0, 0, 0.52778],
    108: [0, 0.69444, 0, 0, 0.27778],
    109: [0, 0.43056, 0, 0, 0.83334],
    110: [0, 0.43056, 0, 0, 0.55556],
    111: [0, 0.43056, 0, 0, 0.5],
    112: [0.19444, 0.43056, 0, 0, 0.55556],
    113: [0.19444, 0.43056, 0, 0, 0.52778],
    114: [0, 0.43056, 0, 0, 0.39167],
    115: [0, 0.43056, 0, 0, 0.39445],
    116: [0, 0.61508, 0, 0, 0.38889],
    117: [0, 0.43056, 0, 0, 0.55556],
    118: [0, 0.43056, 0.01389, 0, 0.52778],
    119: [0, 0.43056, 0.01389, 0, 0.72222],
    120: [0, 0.43056, 0, 0, 0.52778],
    121: [0.19444, 0.43056, 0.01389, 0, 0.52778],
    122: [0, 0.43056, 0, 0, 0.44445],
    123: [0.25, 0.75, 0, 0, 0.5],
    124: [0.25, 0.75, 0, 0, 0.27778],
    125: [0.25, 0.75, 0, 0, 0.5],
    126: [0.35, 0.31786, 0, 0, 0.5],
    160: [0, 0, 0, 0, 0.25],
    163: [0, 0.69444, 0, 0, 0.76909],
    167: [0.19444, 0.69444, 0, 0, 0.44445],
    168: [0, 0.66786, 0, 0, 0.5],
    172: [0, 0.43056, 0, 0, 0.66667],
    176: [0, 0.69444, 0, 0, 0.75],
    177: [0.08333, 0.58333, 0, 0, 0.77778],
    182: [0.19444, 0.69444, 0, 0, 0.61111],
    184: [0.17014, 0, 0, 0, 0.44445],
    198: [0, 0.68333, 0, 0, 0.90278],
    215: [0.08333, 0.58333, 0, 0, 0.77778],
    216: [0.04861, 0.73194, 0, 0, 0.77778],
    223: [0, 0.69444, 0, 0, 0.5],
    230: [0, 0.43056, 0, 0, 0.72222],
    247: [0.08333, 0.58333, 0, 0, 0.77778],
    248: [0.09722, 0.52778, 0, 0, 0.5],
    305: [0, 0.43056, 0, 0, 0.27778],
    338: [0, 0.68333, 0, 0, 1.01389],
    339: [0, 0.43056, 0, 0, 0.77778],
    567: [0.19444, 0.43056, 0, 0, 0.30556],
    710: [0, 0.69444, 0, 0, 0.5],
    711: [0, 0.62847, 0, 0, 0.5],
    713: [0, 0.56778, 0, 0, 0.5],
    714: [0, 0.69444, 0, 0, 0.5],
    715: [0, 0.69444, 0, 0, 0.5],
    728: [0, 0.69444, 0, 0, 0.5],
    729: [0, 0.66786, 0, 0, 0.27778],
    730: [0, 0.69444, 0, 0, 0.75],
    732: [0, 0.66786, 0, 0, 0.5],
    733: [0, 0.69444, 0, 0, 0.5],
    915: [0, 0.68333, 0, 0, 0.625],
    916: [0, 0.68333, 0, 0, 0.83334],
    920: [0, 0.68333, 0, 0, 0.77778],
    923: [0, 0.68333, 0, 0, 0.69445],
    926: [0, 0.68333, 0, 0, 0.66667],
    928: [0, 0.68333, 0, 0, 0.75],
    931: [0, 0.68333, 0, 0, 0.72222],
    933: [0, 0.68333, 0, 0, 0.77778],
    934: [0, 0.68333, 0, 0, 0.72222],
    936: [0, 0.68333, 0, 0, 0.77778],
    937: [0, 0.68333, 0, 0, 0.72222],
    8211: [0, 0.43056, 0.02778, 0, 0.5],
    8212: [0, 0.43056, 0.02778, 0, 1],
    8216: [0, 0.69444, 0, 0, 0.27778],
    8217: [0, 0.69444, 0, 0, 0.27778],
    8220: [0, 0.69444, 0, 0, 0.5],
    8221: [0, 0.69444, 0, 0, 0.5],
    8224: [0.19444, 0.69444, 0, 0, 0.44445],
    8225: [0.19444, 0.69444, 0, 0, 0.44445],
    8230: [0, 0.123, 0, 0, 1.172],
    8242: [0, 0.55556, 0, 0, 0.275],
    8407: [0, 0.71444, 0.15382, 0, 0.5],
    8463: [0, 0.68889, 0, 0, 0.54028],
    8465: [0, 0.69444, 0, 0, 0.72222],
    8467: [0, 0.69444, 0, 0.11111, 0.41667],
    8472: [0.19444, 0.43056, 0, 0.11111, 0.63646],
    8476: [0, 0.69444, 0, 0, 0.72222],
    8501: [0, 0.69444, 0, 0, 0.61111],
    8592: [-0.13313, 0.36687, 0, 0, 1],
    8593: [0.19444, 0.69444, 0, 0, 0.5],
    8594: [-0.13313, 0.36687, 0, 0, 1],
    8595: [0.19444, 0.69444, 0, 0, 0.5],
    8596: [-0.13313, 0.36687, 0, 0, 1],
    8597: [0.25, 0.75, 0, 0, 0.5],
    8598: [0.19444, 0.69444, 0, 0, 1],
    8599: [0.19444, 0.69444, 0, 0, 1],
    8600: [0.19444, 0.69444, 0, 0, 1],
    8601: [0.19444, 0.69444, 0, 0, 1],
    8614: [0.011, 0.511, 0, 0, 1],
    8617: [0.011, 0.511, 0, 0, 1.126],
    8618: [0.011, 0.511, 0, 0, 1.126],
    8636: [-0.13313, 0.36687, 0, 0, 1],
    8637: [-0.13313, 0.36687, 0, 0, 1],
    8640: [-0.13313, 0.36687, 0, 0, 1],
    8641: [-0.13313, 0.36687, 0, 0, 1],
    8652: [0.011, 0.671, 0, 0, 1],
    8656: [-0.13313, 0.36687, 0, 0, 1],
    8657: [0.19444, 0.69444, 0, 0, 0.61111],
    8658: [-0.13313, 0.36687, 0, 0, 1],
    8659: [0.19444, 0.69444, 0, 0, 0.61111],
    8660: [-0.13313, 0.36687, 0, 0, 1],
    8661: [0.25, 0.75, 0, 0, 0.61111],
    8704: [0, 0.69444, 0, 0, 0.55556],
    8706: [0, 0.69444, 0.05556, 0.08334, 0.5309],
    8707: [0, 0.69444, 0, 0, 0.55556],
    8709: [0.05556, 0.75, 0, 0, 0.5],
    8711: [0, 0.68333, 0, 0, 0.83334],
    8712: [0.0391, 0.5391, 0, 0, 0.66667],
    8715: [0.0391, 0.5391, 0, 0, 0.66667],
    8722: [0.08333, 0.58333, 0, 0, 0.77778],
    8723: [0.08333, 0.58333, 0, 0, 0.77778],
    8725: [0.25, 0.75, 0, 0, 0.5],
    8726: [0.25, 0.75, 0, 0, 0.5],
    8727: [-0.03472, 0.46528, 0, 0, 0.5],
    8728: [-0.05555, 0.44445, 0, 0, 0.5],
    8729: [-0.05555, 0.44445, 0, 0, 0.5],
    8730: [0.2, 0.8, 0, 0, 0.83334],
    8733: [0, 0.43056, 0, 0, 0.77778],
    8734: [0, 0.43056, 0, 0, 1],
    8736: [0, 0.69224, 0, 0, 0.72222],
    8739: [0.25, 0.75, 0, 0, 0.27778],
    8741: [0.25, 0.75, 0, 0, 0.5],
    8743: [0, 0.55556, 0, 0, 0.66667],
    8744: [0, 0.55556, 0, 0, 0.66667],
    8745: [0, 0.55556, 0, 0, 0.66667],
    8746: [0, 0.55556, 0, 0, 0.66667],
    8747: [0.19444, 0.69444, 0.11111, 0, 0.41667],
    8764: [-0.13313, 0.36687, 0, 0, 0.77778],
    8768: [0.19444, 0.69444, 0, 0, 0.27778],
    8771: [-0.03625, 0.46375, 0, 0, 0.77778],
    8773: [-0.022, 0.589, 0, 0, 0.778],
    8776: [-0.01688, 0.48312, 0, 0, 0.77778],
    8781: [-0.03625, 0.46375, 0, 0, 0.77778],
    8784: [-0.133, 0.673, 0, 0, 0.778],
    8801: [-0.03625, 0.46375, 0, 0, 0.77778],
    8804: [0.13597, 0.63597, 0, 0, 0.77778],
    8805: [0.13597, 0.63597, 0, 0, 0.77778],
    8810: [0.0391, 0.5391, 0, 0, 1],
    8811: [0.0391, 0.5391, 0, 0, 1],
    8826: [0.0391, 0.5391, 0, 0, 0.77778],
    8827: [0.0391, 0.5391, 0, 0, 0.77778],
    8834: [0.0391, 0.5391, 0, 0, 0.77778],
    8835: [0.0391, 0.5391, 0, 0, 0.77778],
    8838: [0.13597, 0.63597, 0, 0, 0.77778],
    8839: [0.13597, 0.63597, 0, 0, 0.77778],
    8846: [0, 0.55556, 0, 0, 0.66667],
    8849: [0.13597, 0.63597, 0, 0, 0.77778],
    8850: [0.13597, 0.63597, 0, 0, 0.77778],
    8851: [0, 0.55556, 0, 0, 0.66667],
    8852: [0, 0.55556, 0, 0, 0.66667],
    8853: [0.08333, 0.58333, 0, 0, 0.77778],
    8854: [0.08333, 0.58333, 0, 0, 0.77778],
    8855: [0.08333, 0.58333, 0, 0, 0.77778],
    8856: [0.08333, 0.58333, 0, 0, 0.77778],
    8857: [0.08333, 0.58333, 0, 0, 0.77778],
    8866: [0, 0.69444, 0, 0, 0.61111],
    8867: [0, 0.69444, 0, 0, 0.61111],
    8868: [0, 0.69444, 0, 0, 0.77778],
    8869: [0, 0.69444, 0, 0, 0.77778],
    8872: [0.249, 0.75, 0, 0, 0.867],
    8900: [-0.05555, 0.44445, 0, 0, 0.5],
    8901: [-0.05555, 0.44445, 0, 0, 0.27778],
    8902: [-0.03472, 0.46528, 0, 0, 0.5],
    8904: [5e-3, 0.505, 0, 0, 0.9],
    8942: [0.03, 0.903, 0, 0, 0.278],
    8943: [-0.19, 0.313, 0, 0, 1.172],
    8945: [-0.1, 0.823, 0, 0, 1.282],
    8968: [0.25, 0.75, 0, 0, 0.44445],
    8969: [0.25, 0.75, 0, 0, 0.44445],
    8970: [0.25, 0.75, 0, 0, 0.44445],
    8971: [0.25, 0.75, 0, 0, 0.44445],
    8994: [-0.14236, 0.35764, 0, 0, 1],
    8995: [-0.14236, 0.35764, 0, 0, 1],
    9136: [0.244, 0.744, 0, 0, 0.412],
    9137: [0.244, 0.745, 0, 0, 0.412],
    9651: [0.19444, 0.69444, 0, 0, 0.88889],
    9657: [-0.03472, 0.46528, 0, 0, 0.5],
    9661: [0.19444, 0.69444, 0, 0, 0.88889],
    9667: [-0.03472, 0.46528, 0, 0, 0.5],
    9711: [0.19444, 0.69444, 0, 0, 1],
    9824: [0.12963, 0.69444, 0, 0, 0.77778],
    9825: [0.12963, 0.69444, 0, 0, 0.77778],
    9826: [0.12963, 0.69444, 0, 0, 0.77778],
    9827: [0.12963, 0.69444, 0, 0, 0.77778],
    9837: [0, 0.75, 0, 0, 0.38889],
    9838: [0.19444, 0.69444, 0, 0, 0.38889],
    9839: [0.19444, 0.69444, 0, 0, 0.38889],
    10216: [0.25, 0.75, 0, 0, 0.38889],
    10217: [0.25, 0.75, 0, 0, 0.38889],
    10222: [0.244, 0.744, 0, 0, 0.412],
    10223: [0.244, 0.745, 0, 0, 0.412],
    10229: [0.011, 0.511, 0, 0, 1.609],
    10230: [0.011, 0.511, 0, 0, 1.638],
    10231: [0.011, 0.511, 0, 0, 1.859],
    10232: [0.024, 0.525, 0, 0, 1.609],
    10233: [0.024, 0.525, 0, 0, 1.638],
    10234: [0.024, 0.525, 0, 0, 1.858],
    10236: [0.011, 0.511, 0, 0, 1.638],
    10815: [0, 0.68333, 0, 0, 0.75],
    10927: [0.13597, 0.63597, 0, 0, 0.77778],
    10928: [0.13597, 0.63597, 0, 0, 0.77778],
    57376: [0.19444, 0.69444, 0, 0, 0]
  },
  "Math-BoldItalic": {
    32: [0, 0, 0, 0, 0.25],
    48: [0, 0.44444, 0, 0, 0.575],
    49: [0, 0.44444, 0, 0, 0.575],
    50: [0, 0.44444, 0, 0, 0.575],
    51: [0.19444, 0.44444, 0, 0, 0.575],
    52: [0.19444, 0.44444, 0, 0, 0.575],
    53: [0.19444, 0.44444, 0, 0, 0.575],
    54: [0, 0.64444, 0, 0, 0.575],
    55: [0.19444, 0.44444, 0, 0, 0.575],
    56: [0, 0.64444, 0, 0, 0.575],
    57: [0.19444, 0.44444, 0, 0, 0.575],
    65: [0, 0.68611, 0, 0, 0.86944],
    66: [0, 0.68611, 0.04835, 0, 0.8664],
    67: [0, 0.68611, 0.06979, 0, 0.81694],
    68: [0, 0.68611, 0.03194, 0, 0.93812],
    69: [0, 0.68611, 0.05451, 0, 0.81007],
    70: [0, 0.68611, 0.15972, 0, 0.68889],
    71: [0, 0.68611, 0, 0, 0.88673],
    72: [0, 0.68611, 0.08229, 0, 0.98229],
    73: [0, 0.68611, 0.07778, 0, 0.51111],
    74: [0, 0.68611, 0.10069, 0, 0.63125],
    75: [0, 0.68611, 0.06979, 0, 0.97118],
    76: [0, 0.68611, 0, 0, 0.75555],
    77: [0, 0.68611, 0.11424, 0, 1.14201],
    78: [0, 0.68611, 0.11424, 0, 0.95034],
    79: [0, 0.68611, 0.03194, 0, 0.83666],
    80: [0, 0.68611, 0.15972, 0, 0.72309],
    81: [0.19444, 0.68611, 0, 0, 0.86861],
    82: [0, 0.68611, 421e-5, 0, 0.87235],
    83: [0, 0.68611, 0.05382, 0, 0.69271],
    84: [0, 0.68611, 0.15972, 0, 0.63663],
    85: [0, 0.68611, 0.11424, 0, 0.80027],
    86: [0, 0.68611, 0.25555, 0, 0.67778],
    87: [0, 0.68611, 0.15972, 0, 1.09305],
    88: [0, 0.68611, 0.07778, 0, 0.94722],
    89: [0, 0.68611, 0.25555, 0, 0.67458],
    90: [0, 0.68611, 0.06979, 0, 0.77257],
    97: [0, 0.44444, 0, 0, 0.63287],
    98: [0, 0.69444, 0, 0, 0.52083],
    99: [0, 0.44444, 0, 0, 0.51342],
    100: [0, 0.69444, 0, 0, 0.60972],
    101: [0, 0.44444, 0, 0, 0.55361],
    102: [0.19444, 0.69444, 0.11042, 0, 0.56806],
    103: [0.19444, 0.44444, 0.03704, 0, 0.5449],
    104: [0, 0.69444, 0, 0, 0.66759],
    105: [0, 0.69326, 0, 0, 0.4048],
    106: [0.19444, 0.69326, 0.0622, 0, 0.47083],
    107: [0, 0.69444, 0.01852, 0, 0.6037],
    108: [0, 0.69444, 88e-4, 0, 0.34815],
    109: [0, 0.44444, 0, 0, 1.0324],
    110: [0, 0.44444, 0, 0, 0.71296],
    111: [0, 0.44444, 0, 0, 0.58472],
    112: [0.19444, 0.44444, 0, 0, 0.60092],
    113: [0.19444, 0.44444, 0.03704, 0, 0.54213],
    114: [0, 0.44444, 0.03194, 0, 0.5287],
    115: [0, 0.44444, 0, 0, 0.53125],
    116: [0, 0.63492, 0, 0, 0.41528],
    117: [0, 0.44444, 0, 0, 0.68102],
    118: [0, 0.44444, 0.03704, 0, 0.56666],
    119: [0, 0.44444, 0.02778, 0, 0.83148],
    120: [0, 0.44444, 0, 0, 0.65903],
    121: [0.19444, 0.44444, 0.03704, 0, 0.59028],
    122: [0, 0.44444, 0.04213, 0, 0.55509],
    160: [0, 0, 0, 0, 0.25],
    915: [0, 0.68611, 0.15972, 0, 0.65694],
    916: [0, 0.68611, 0, 0, 0.95833],
    920: [0, 0.68611, 0.03194, 0, 0.86722],
    923: [0, 0.68611, 0, 0, 0.80555],
    926: [0, 0.68611, 0.07458, 0, 0.84125],
    928: [0, 0.68611, 0.08229, 0, 0.98229],
    931: [0, 0.68611, 0.05451, 0, 0.88507],
    933: [0, 0.68611, 0.15972, 0, 0.67083],
    934: [0, 0.68611, 0, 0, 0.76666],
    936: [0, 0.68611, 0.11653, 0, 0.71402],
    937: [0, 0.68611, 0.04835, 0, 0.8789],
    945: [0, 0.44444, 0, 0, 0.76064],
    946: [0.19444, 0.69444, 0.03403, 0, 0.65972],
    947: [0.19444, 0.44444, 0.06389, 0, 0.59003],
    948: [0, 0.69444, 0.03819, 0, 0.52222],
    949: [0, 0.44444, 0, 0, 0.52882],
    950: [0.19444, 0.69444, 0.06215, 0, 0.50833],
    951: [0.19444, 0.44444, 0.03704, 0, 0.6],
    952: [0, 0.69444, 0.03194, 0, 0.5618],
    953: [0, 0.44444, 0, 0, 0.41204],
    954: [0, 0.44444, 0, 0, 0.66759],
    955: [0, 0.69444, 0, 0, 0.67083],
    956: [0.19444, 0.44444, 0, 0, 0.70787],
    957: [0, 0.44444, 0.06898, 0, 0.57685],
    958: [0.19444, 0.69444, 0.03021, 0, 0.50833],
    959: [0, 0.44444, 0, 0, 0.58472],
    960: [0, 0.44444, 0.03704, 0, 0.68241],
    961: [0.19444, 0.44444, 0, 0, 0.6118],
    962: [0.09722, 0.44444, 0.07917, 0, 0.42361],
    963: [0, 0.44444, 0.03704, 0, 0.68588],
    964: [0, 0.44444, 0.13472, 0, 0.52083],
    965: [0, 0.44444, 0.03704, 0, 0.63055],
    966: [0.19444, 0.44444, 0, 0, 0.74722],
    967: [0.19444, 0.44444, 0, 0, 0.71805],
    968: [0.19444, 0.69444, 0.03704, 0, 0.75833],
    969: [0, 0.44444, 0.03704, 0, 0.71782],
    977: [0, 0.69444, 0, 0, 0.69155],
    981: [0.19444, 0.69444, 0, 0, 0.7125],
    982: [0, 0.44444, 0.03194, 0, 0.975],
    1009: [0.19444, 0.44444, 0, 0, 0.6118],
    1013: [0, 0.44444, 0, 0, 0.48333],
    57649: [0, 0.44444, 0, 0, 0.39352],
    57911: [0.19444, 0.44444, 0, 0, 0.43889]
  },
  "Math-Italic": {
    32: [0, 0, 0, 0, 0.25],
    48: [0, 0.43056, 0, 0, 0.5],
    49: [0, 0.43056, 0, 0, 0.5],
    50: [0, 0.43056, 0, 0, 0.5],
    51: [0.19444, 0.43056, 0, 0, 0.5],
    52: [0.19444, 0.43056, 0, 0, 0.5],
    53: [0.19444, 0.43056, 0, 0, 0.5],
    54: [0, 0.64444, 0, 0, 0.5],
    55: [0.19444, 0.43056, 0, 0, 0.5],
    56: [0, 0.64444, 0, 0, 0.5],
    57: [0.19444, 0.43056, 0, 0, 0.5],
    65: [0, 0.68333, 0, 0.13889, 0.75],
    66: [0, 0.68333, 0.05017, 0.08334, 0.75851],
    67: [0, 0.68333, 0.07153, 0.08334, 0.71472],
    68: [0, 0.68333, 0.02778, 0.05556, 0.82792],
    69: [0, 0.68333, 0.05764, 0.08334, 0.7382],
    70: [0, 0.68333, 0.13889, 0.08334, 0.64306],
    71: [0, 0.68333, 0, 0.08334, 0.78625],
    72: [0, 0.68333, 0.08125, 0.05556, 0.83125],
    73: [0, 0.68333, 0.07847, 0.11111, 0.43958],
    74: [0, 0.68333, 0.09618, 0.16667, 0.55451],
    75: [0, 0.68333, 0.07153, 0.05556, 0.84931],
    76: [0, 0.68333, 0, 0.02778, 0.68056],
    77: [0, 0.68333, 0.10903, 0.08334, 0.97014],
    78: [0, 0.68333, 0.10903, 0.08334, 0.80347],
    79: [0, 0.68333, 0.02778, 0.08334, 0.76278],
    80: [0, 0.68333, 0.13889, 0.08334, 0.64201],
    81: [0.19444, 0.68333, 0, 0.08334, 0.79056],
    82: [0, 0.68333, 773e-5, 0.08334, 0.75929],
    83: [0, 0.68333, 0.05764, 0.08334, 0.6132],
    84: [0, 0.68333, 0.13889, 0.08334, 0.58438],
    85: [0, 0.68333, 0.10903, 0.02778, 0.68278],
    86: [0, 0.68333, 0.22222, 0, 0.58333],
    87: [0, 0.68333, 0.13889, 0, 0.94445],
    88: [0, 0.68333, 0.07847, 0.08334, 0.82847],
    89: [0, 0.68333, 0.22222, 0, 0.58056],
    90: [0, 0.68333, 0.07153, 0.08334, 0.68264],
    97: [0, 0.43056, 0, 0, 0.52859],
    98: [0, 0.69444, 0, 0, 0.42917],
    99: [0, 0.43056, 0, 0.05556, 0.43276],
    100: [0, 0.69444, 0, 0.16667, 0.52049],
    101: [0, 0.43056, 0, 0.05556, 0.46563],
    102: [0.19444, 0.69444, 0.10764, 0.16667, 0.48959],
    103: [0.19444, 0.43056, 0.03588, 0.02778, 0.47697],
    104: [0, 0.69444, 0, 0, 0.57616],
    105: [0, 0.65952, 0, 0, 0.34451],
    106: [0.19444, 0.65952, 0.05724, 0, 0.41181],
    107: [0, 0.69444, 0.03148, 0, 0.5206],
    108: [0, 0.69444, 0.01968, 0.08334, 0.29838],
    109: [0, 0.43056, 0, 0, 0.87801],
    110: [0, 0.43056, 0, 0, 0.60023],
    111: [0, 0.43056, 0, 0.05556, 0.48472],
    112: [0.19444, 0.43056, 0, 0.08334, 0.50313],
    113: [0.19444, 0.43056, 0.03588, 0.08334, 0.44641],
    114: [0, 0.43056, 0.02778, 0.05556, 0.45116],
    115: [0, 0.43056, 0, 0.05556, 0.46875],
    116: [0, 0.61508, 0, 0.08334, 0.36111],
    117: [0, 0.43056, 0, 0.02778, 0.57246],
    118: [0, 0.43056, 0.03588, 0.02778, 0.48472],
    119: [0, 0.43056, 0.02691, 0.08334, 0.71592],
    120: [0, 0.43056, 0, 0.02778, 0.57153],
    121: [0.19444, 0.43056, 0.03588, 0.05556, 0.49028],
    122: [0, 0.43056, 0.04398, 0.05556, 0.46505],
    160: [0, 0, 0, 0, 0.25],
    915: [0, 0.68333, 0.13889, 0.08334, 0.61528],
    916: [0, 0.68333, 0, 0.16667, 0.83334],
    920: [0, 0.68333, 0.02778, 0.08334, 0.76278],
    923: [0, 0.68333, 0, 0.16667, 0.69445],
    926: [0, 0.68333, 0.07569, 0.08334, 0.74236],
    928: [0, 0.68333, 0.08125, 0.05556, 0.83125],
    931: [0, 0.68333, 0.05764, 0.08334, 0.77986],
    933: [0, 0.68333, 0.13889, 0.05556, 0.58333],
    934: [0, 0.68333, 0, 0.08334, 0.66667],
    936: [0, 0.68333, 0.11, 0.05556, 0.61222],
    937: [0, 0.68333, 0.05017, 0.08334, 0.7724],
    945: [0, 0.43056, 37e-4, 0.02778, 0.6397],
    946: [0.19444, 0.69444, 0.05278, 0.08334, 0.56563],
    947: [0.19444, 0.43056, 0.05556, 0, 0.51773],
    948: [0, 0.69444, 0.03785, 0.05556, 0.44444],
    949: [0, 0.43056, 0, 0.08334, 0.46632],
    950: [0.19444, 0.69444, 0.07378, 0.08334, 0.4375],
    951: [0.19444, 0.43056, 0.03588, 0.05556, 0.49653],
    952: [0, 0.69444, 0.02778, 0.08334, 0.46944],
    953: [0, 0.43056, 0, 0.05556, 0.35394],
    954: [0, 0.43056, 0, 0, 0.57616],
    955: [0, 0.69444, 0, 0, 0.58334],
    956: [0.19444, 0.43056, 0, 0.02778, 0.60255],
    957: [0, 0.43056, 0.06366, 0.02778, 0.49398],
    958: [0.19444, 0.69444, 0.04601, 0.11111, 0.4375],
    959: [0, 0.43056, 0, 0.05556, 0.48472],
    960: [0, 0.43056, 0.03588, 0, 0.57003],
    961: [0.19444, 0.43056, 0, 0.08334, 0.51702],
    962: [0.09722, 0.43056, 0.07986, 0.08334, 0.36285],
    963: [0, 0.43056, 0.03588, 0, 0.57141],
    964: [0, 0.43056, 0.1132, 0.02778, 0.43715],
    965: [0, 0.43056, 0.03588, 0.02778, 0.54028],
    966: [0.19444, 0.43056, 0, 0.08334, 0.65417],
    967: [0.19444, 0.43056, 0, 0.05556, 0.62569],
    968: [0.19444, 0.69444, 0.03588, 0.11111, 0.65139],
    969: [0, 0.43056, 0.03588, 0, 0.62245],
    977: [0, 0.69444, 0, 0.08334, 0.59144],
    981: [0.19444, 0.69444, 0, 0.08334, 0.59583],
    982: [0, 0.43056, 0.02778, 0, 0.82813],
    1009: [0.19444, 0.43056, 0, 0.08334, 0.51702],
    1013: [0, 0.43056, 0, 0.05556, 0.4059],
    57649: [0, 0.43056, 0, 0.02778, 0.32246],
    57911: [0.19444, 0.43056, 0, 0.08334, 0.38403]
  },
  "SansSerif-Bold": {
    32: [0, 0, 0, 0, 0.25],
    33: [0, 0.69444, 0, 0, 0.36667],
    34: [0, 0.69444, 0, 0, 0.55834],
    35: [0.19444, 0.69444, 0, 0, 0.91667],
    36: [0.05556, 0.75, 0, 0, 0.55],
    37: [0.05556, 0.75, 0, 0, 1.02912],
    38: [0, 0.69444, 0, 0, 0.83056],
    39: [0, 0.69444, 0, 0, 0.30556],
    40: [0.25, 0.75, 0, 0, 0.42778],
    41: [0.25, 0.75, 0, 0, 0.42778],
    42: [0, 0.75, 0, 0, 0.55],
    43: [0.11667, 0.61667, 0, 0, 0.85556],
    44: [0.10556, 0.13056, 0, 0, 0.30556],
    45: [0, 0.45833, 0, 0, 0.36667],
    46: [0, 0.13056, 0, 0, 0.30556],
    47: [0.25, 0.75, 0, 0, 0.55],
    48: [0, 0.69444, 0, 0, 0.55],
    49: [0, 0.69444, 0, 0, 0.55],
    50: [0, 0.69444, 0, 0, 0.55],
    51: [0, 0.69444, 0, 0, 0.55],
    52: [0, 0.69444, 0, 0, 0.55],
    53: [0, 0.69444, 0, 0, 0.55],
    54: [0, 0.69444, 0, 0, 0.55],
    55: [0, 0.69444, 0, 0, 0.55],
    56: [0, 0.69444, 0, 0, 0.55],
    57: [0, 0.69444, 0, 0, 0.55],
    58: [0, 0.45833, 0, 0, 0.30556],
    59: [0.10556, 0.45833, 0, 0, 0.30556],
    61: [-0.09375, 0.40625, 0, 0, 0.85556],
    63: [0, 0.69444, 0, 0, 0.51945],
    64: [0, 0.69444, 0, 0, 0.73334],
    65: [0, 0.69444, 0, 0, 0.73334],
    66: [0, 0.69444, 0, 0, 0.73334],
    67: [0, 0.69444, 0, 0, 0.70278],
    68: [0, 0.69444, 0, 0, 0.79445],
    69: [0, 0.69444, 0, 0, 0.64167],
    70: [0, 0.69444, 0, 0, 0.61111],
    71: [0, 0.69444, 0, 0, 0.73334],
    72: [0, 0.69444, 0, 0, 0.79445],
    73: [0, 0.69444, 0, 0, 0.33056],
    74: [0, 0.69444, 0, 0, 0.51945],
    75: [0, 0.69444, 0, 0, 0.76389],
    76: [0, 0.69444, 0, 0, 0.58056],
    77: [0, 0.69444, 0, 0, 0.97778],
    78: [0, 0.69444, 0, 0, 0.79445],
    79: [0, 0.69444, 0, 0, 0.79445],
    80: [0, 0.69444, 0, 0, 0.70278],
    81: [0.10556, 0.69444, 0, 0, 0.79445],
    82: [0, 0.69444, 0, 0, 0.70278],
    83: [0, 0.69444, 0, 0, 0.61111],
    84: [0, 0.69444, 0, 0, 0.73334],
    85: [0, 0.69444, 0, 0, 0.76389],
    86: [0, 0.69444, 0.01528, 0, 0.73334],
    87: [0, 0.69444, 0.01528, 0, 1.03889],
    88: [0, 0.69444, 0, 0, 0.73334],
    89: [0, 0.69444, 0.0275, 0, 0.73334],
    90: [0, 0.69444, 0, 0, 0.67223],
    91: [0.25, 0.75, 0, 0, 0.34306],
    93: [0.25, 0.75, 0, 0, 0.34306],
    94: [0, 0.69444, 0, 0, 0.55],
    95: [0.35, 0.10833, 0.03056, 0, 0.55],
    97: [0, 0.45833, 0, 0, 0.525],
    98: [0, 0.69444, 0, 0, 0.56111],
    99: [0, 0.45833, 0, 0, 0.48889],
    100: [0, 0.69444, 0, 0, 0.56111],
    101: [0, 0.45833, 0, 0, 0.51111],
    102: [0, 0.69444, 0.07639, 0, 0.33611],
    103: [0.19444, 0.45833, 0.01528, 0, 0.55],
    104: [0, 0.69444, 0, 0, 0.56111],
    105: [0, 0.69444, 0, 0, 0.25556],
    106: [0.19444, 0.69444, 0, 0, 0.28611],
    107: [0, 0.69444, 0, 0, 0.53056],
    108: [0, 0.69444, 0, 0, 0.25556],
    109: [0, 0.45833, 0, 0, 0.86667],
    110: [0, 0.45833, 0, 0, 0.56111],
    111: [0, 0.45833, 0, 0, 0.55],
    112: [0.19444, 0.45833, 0, 0, 0.56111],
    113: [0.19444, 0.45833, 0, 0, 0.56111],
    114: [0, 0.45833, 0.01528, 0, 0.37222],
    115: [0, 0.45833, 0, 0, 0.42167],
    116: [0, 0.58929, 0, 0, 0.40417],
    117: [0, 0.45833, 0, 0, 0.56111],
    118: [0, 0.45833, 0.01528, 0, 0.5],
    119: [0, 0.45833, 0.01528, 0, 0.74445],
    120: [0, 0.45833, 0, 0, 0.5],
    121: [0.19444, 0.45833, 0.01528, 0, 0.5],
    122: [0, 0.45833, 0, 0, 0.47639],
    126: [0.35, 0.34444, 0, 0, 0.55],
    160: [0, 0, 0, 0, 0.25],
    168: [0, 0.69444, 0, 0, 0.55],
    176: [0, 0.69444, 0, 0, 0.73334],
    180: [0, 0.69444, 0, 0, 0.55],
    184: [0.17014, 0, 0, 0, 0.48889],
    305: [0, 0.45833, 0, 0, 0.25556],
    567: [0.19444, 0.45833, 0, 0, 0.28611],
    710: [0, 0.69444, 0, 0, 0.55],
    711: [0, 0.63542, 0, 0, 0.55],
    713: [0, 0.63778, 0, 0, 0.55],
    728: [0, 0.69444, 0, 0, 0.55],
    729: [0, 0.69444, 0, 0, 0.30556],
    730: [0, 0.69444, 0, 0, 0.73334],
    732: [0, 0.69444, 0, 0, 0.55],
    733: [0, 0.69444, 0, 0, 0.55],
    915: [0, 0.69444, 0, 0, 0.58056],
    916: [0, 0.69444, 0, 0, 0.91667],
    920: [0, 0.69444, 0, 0, 0.85556],
    923: [0, 0.69444, 0, 0, 0.67223],
    926: [0, 0.69444, 0, 0, 0.73334],
    928: [0, 0.69444, 0, 0, 0.79445],
    931: [0, 0.69444, 0, 0, 0.79445],
    933: [0, 0.69444, 0, 0, 0.85556],
    934: [0, 0.69444, 0, 0, 0.79445],
    936: [0, 0.69444, 0, 0, 0.85556],
    937: [0, 0.69444, 0, 0, 0.79445],
    8211: [0, 0.45833, 0.03056, 0, 0.55],
    8212: [0, 0.45833, 0.03056, 0, 1.10001],
    8216: [0, 0.69444, 0, 0, 0.30556],
    8217: [0, 0.69444, 0, 0, 0.30556],
    8220: [0, 0.69444, 0, 0, 0.55834],
    8221: [0, 0.69444, 0, 0, 0.55834]
  },
  "SansSerif-Italic": {
    32: [0, 0, 0, 0, 0.25],
    33: [0, 0.69444, 0.05733, 0, 0.31945],
    34: [0, 0.69444, 316e-5, 0, 0.5],
    35: [0.19444, 0.69444, 0.05087, 0, 0.83334],
    36: [0.05556, 0.75, 0.11156, 0, 0.5],
    37: [0.05556, 0.75, 0.03126, 0, 0.83334],
    38: [0, 0.69444, 0.03058, 0, 0.75834],
    39: [0, 0.69444, 0.07816, 0, 0.27778],
    40: [0.25, 0.75, 0.13164, 0, 0.38889],
    41: [0.25, 0.75, 0.02536, 0, 0.38889],
    42: [0, 0.75, 0.11775, 0, 0.5],
    43: [0.08333, 0.58333, 0.02536, 0, 0.77778],
    44: [0.125, 0.08333, 0, 0, 0.27778],
    45: [0, 0.44444, 0.01946, 0, 0.33333],
    46: [0, 0.08333, 0, 0, 0.27778],
    47: [0.25, 0.75, 0.13164, 0, 0.5],
    48: [0, 0.65556, 0.11156, 0, 0.5],
    49: [0, 0.65556, 0.11156, 0, 0.5],
    50: [0, 0.65556, 0.11156, 0, 0.5],
    51: [0, 0.65556, 0.11156, 0, 0.5],
    52: [0, 0.65556, 0.11156, 0, 0.5],
    53: [0, 0.65556, 0.11156, 0, 0.5],
    54: [0, 0.65556, 0.11156, 0, 0.5],
    55: [0, 0.65556, 0.11156, 0, 0.5],
    56: [0, 0.65556, 0.11156, 0, 0.5],
    57: [0, 0.65556, 0.11156, 0, 0.5],
    58: [0, 0.44444, 0.02502, 0, 0.27778],
    59: [0.125, 0.44444, 0.02502, 0, 0.27778],
    61: [-0.13, 0.37, 0.05087, 0, 0.77778],
    63: [0, 0.69444, 0.11809, 0, 0.47222],
    64: [0, 0.69444, 0.07555, 0, 0.66667],
    65: [0, 0.69444, 0, 0, 0.66667],
    66: [0, 0.69444, 0.08293, 0, 0.66667],
    67: [0, 0.69444, 0.11983, 0, 0.63889],
    68: [0, 0.69444, 0.07555, 0, 0.72223],
    69: [0, 0.69444, 0.11983, 0, 0.59722],
    70: [0, 0.69444, 0.13372, 0, 0.56945],
    71: [0, 0.69444, 0.11983, 0, 0.66667],
    72: [0, 0.69444, 0.08094, 0, 0.70834],
    73: [0, 0.69444, 0.13372, 0, 0.27778],
    74: [0, 0.69444, 0.08094, 0, 0.47222],
    75: [0, 0.69444, 0.11983, 0, 0.69445],
    76: [0, 0.69444, 0, 0, 0.54167],
    77: [0, 0.69444, 0.08094, 0, 0.875],
    78: [0, 0.69444, 0.08094, 0, 0.70834],
    79: [0, 0.69444, 0.07555, 0, 0.73611],
    80: [0, 0.69444, 0.08293, 0, 0.63889],
    81: [0.125, 0.69444, 0.07555, 0, 0.73611],
    82: [0, 0.69444, 0.08293, 0, 0.64584],
    83: [0, 0.69444, 0.09205, 0, 0.55556],
    84: [0, 0.69444, 0.13372, 0, 0.68056],
    85: [0, 0.69444, 0.08094, 0, 0.6875],
    86: [0, 0.69444, 0.1615, 0, 0.66667],
    87: [0, 0.69444, 0.1615, 0, 0.94445],
    88: [0, 0.69444, 0.13372, 0, 0.66667],
    89: [0, 0.69444, 0.17261, 0, 0.66667],
    90: [0, 0.69444, 0.11983, 0, 0.61111],
    91: [0.25, 0.75, 0.15942, 0, 0.28889],
    93: [0.25, 0.75, 0.08719, 0, 0.28889],
    94: [0, 0.69444, 0.0799, 0, 0.5],
    95: [0.35, 0.09444, 0.08616, 0, 0.5],
    97: [0, 0.44444, 981e-5, 0, 0.48056],
    98: [0, 0.69444, 0.03057, 0, 0.51667],
    99: [0, 0.44444, 0.08336, 0, 0.44445],
    100: [0, 0.69444, 0.09483, 0, 0.51667],
    101: [0, 0.44444, 0.06778, 0, 0.44445],
    102: [0, 0.69444, 0.21705, 0, 0.30556],
    103: [0.19444, 0.44444, 0.10836, 0, 0.5],
    104: [0, 0.69444, 0.01778, 0, 0.51667],
    105: [0, 0.67937, 0.09718, 0, 0.23889],
    106: [0.19444, 0.67937, 0.09162, 0, 0.26667],
    107: [0, 0.69444, 0.08336, 0, 0.48889],
    108: [0, 0.69444, 0.09483, 0, 0.23889],
    109: [0, 0.44444, 0.01778, 0, 0.79445],
    110: [0, 0.44444, 0.01778, 0, 0.51667],
    111: [0, 0.44444, 0.06613, 0, 0.5],
    112: [0.19444, 0.44444, 0.0389, 0, 0.51667],
    113: [0.19444, 0.44444, 0.04169, 0, 0.51667],
    114: [0, 0.44444, 0.10836, 0, 0.34167],
    115: [0, 0.44444, 0.0778, 0, 0.38333],
    116: [0, 0.57143, 0.07225, 0, 0.36111],
    117: [0, 0.44444, 0.04169, 0, 0.51667],
    118: [0, 0.44444, 0.10836, 0, 0.46111],
    119: [0, 0.44444, 0.10836, 0, 0.68334],
    120: [0, 0.44444, 0.09169, 0, 0.46111],
    121: [0.19444, 0.44444, 0.10836, 0, 0.46111],
    122: [0, 0.44444, 0.08752, 0, 0.43472],
    126: [0.35, 0.32659, 0.08826, 0, 0.5],
    160: [0, 0, 0, 0, 0.25],
    168: [0, 0.67937, 0.06385, 0, 0.5],
    176: [0, 0.69444, 0, 0, 0.73752],
    184: [0.17014, 0, 0, 0, 0.44445],
    305: [0, 0.44444, 0.04169, 0, 0.23889],
    567: [0.19444, 0.44444, 0.04169, 0, 0.26667],
    710: [0, 0.69444, 0.0799, 0, 0.5],
    711: [0, 0.63194, 0.08432, 0, 0.5],
    713: [0, 0.60889, 0.08776, 0, 0.5],
    714: [0, 0.69444, 0.09205, 0, 0.5],
    715: [0, 0.69444, 0, 0, 0.5],
    728: [0, 0.69444, 0.09483, 0, 0.5],
    729: [0, 0.67937, 0.07774, 0, 0.27778],
    730: [0, 0.69444, 0, 0, 0.73752],
    732: [0, 0.67659, 0.08826, 0, 0.5],
    733: [0, 0.69444, 0.09205, 0, 0.5],
    915: [0, 0.69444, 0.13372, 0, 0.54167],
    916: [0, 0.69444, 0, 0, 0.83334],
    920: [0, 0.69444, 0.07555, 0, 0.77778],
    923: [0, 0.69444, 0, 0, 0.61111],
    926: [0, 0.69444, 0.12816, 0, 0.66667],
    928: [0, 0.69444, 0.08094, 0, 0.70834],
    931: [0, 0.69444, 0.11983, 0, 0.72222],
    933: [0, 0.69444, 0.09031, 0, 0.77778],
    934: [0, 0.69444, 0.04603, 0, 0.72222],
    936: [0, 0.69444, 0.09031, 0, 0.77778],
    937: [0, 0.69444, 0.08293, 0, 0.72222],
    8211: [0, 0.44444, 0.08616, 0, 0.5],
    8212: [0, 0.44444, 0.08616, 0, 1],
    8216: [0, 0.69444, 0.07816, 0, 0.27778],
    8217: [0, 0.69444, 0.07816, 0, 0.27778],
    8220: [0, 0.69444, 0.14205, 0, 0.5],
    8221: [0, 0.69444, 316e-5, 0, 0.5]
  },
  "SansSerif-Regular": {
    32: [0, 0, 0, 0, 0.25],
    33: [0, 0.69444, 0, 0, 0.31945],
    34: [0, 0.69444, 0, 0, 0.5],
    35: [0.19444, 0.69444, 0, 0, 0.83334],
    36: [0.05556, 0.75, 0, 0, 0.5],
    37: [0.05556, 0.75, 0, 0, 0.83334],
    38: [0, 0.69444, 0, 0, 0.75834],
    39: [0, 0.69444, 0, 0, 0.27778],
    40: [0.25, 0.75, 0, 0, 0.38889],
    41: [0.25, 0.75, 0, 0, 0.38889],
    42: [0, 0.75, 0, 0, 0.5],
    43: [0.08333, 0.58333, 0, 0, 0.77778],
    44: [0.125, 0.08333, 0, 0, 0.27778],
    45: [0, 0.44444, 0, 0, 0.33333],
    46: [0, 0.08333, 0, 0, 0.27778],
    47: [0.25, 0.75, 0, 0, 0.5],
    48: [0, 0.65556, 0, 0, 0.5],
    49: [0, 0.65556, 0, 0, 0.5],
    50: [0, 0.65556, 0, 0, 0.5],
    51: [0, 0.65556, 0, 0, 0.5],
    52: [0, 0.65556, 0, 0, 0.5],
    53: [0, 0.65556, 0, 0, 0.5],
    54: [0, 0.65556, 0, 0, 0.5],
    55: [0, 0.65556, 0, 0, 0.5],
    56: [0, 0.65556, 0, 0, 0.5],
    57: [0, 0.65556, 0, 0, 0.5],
    58: [0, 0.44444, 0, 0, 0.27778],
    59: [0.125, 0.44444, 0, 0, 0.27778],
    61: [-0.13, 0.37, 0, 0, 0.77778],
    63: [0, 0.69444, 0, 0, 0.47222],
    64: [0, 0.69444, 0, 0, 0.66667],
    65: [0, 0.69444, 0, 0, 0.66667],
    66: [0, 0.69444, 0, 0, 0.66667],
    67: [0, 0.69444, 0, 0, 0.63889],
    68: [0, 0.69444, 0, 0, 0.72223],
    69: [0, 0.69444, 0, 0, 0.59722],
    70: [0, 0.69444, 0, 0, 0.56945],
    71: [0, 0.69444, 0, 0, 0.66667],
    72: [0, 0.69444, 0, 0, 0.70834],
    73: [0, 0.69444, 0, 0, 0.27778],
    74: [0, 0.69444, 0, 0, 0.47222],
    75: [0, 0.69444, 0, 0, 0.69445],
    76: [0, 0.69444, 0, 0, 0.54167],
    77: [0, 0.69444, 0, 0, 0.875],
    78: [0, 0.69444, 0, 0, 0.70834],
    79: [0, 0.69444, 0, 0, 0.73611],
    80: [0, 0.69444, 0, 0, 0.63889],
    81: [0.125, 0.69444, 0, 0, 0.73611],
    82: [0, 0.69444, 0, 0, 0.64584],
    83: [0, 0.69444, 0, 0, 0.55556],
    84: [0, 0.69444, 0, 0, 0.68056],
    85: [0, 0.69444, 0, 0, 0.6875],
    86: [0, 0.69444, 0.01389, 0, 0.66667],
    87: [0, 0.69444, 0.01389, 0, 0.94445],
    88: [0, 0.69444, 0, 0, 0.66667],
    89: [0, 0.69444, 0.025, 0, 0.66667],
    90: [0, 0.69444, 0, 0, 0.61111],
    91: [0.25, 0.75, 0, 0, 0.28889],
    93: [0.25, 0.75, 0, 0, 0.28889],
    94: [0, 0.69444, 0, 0, 0.5],
    95: [0.35, 0.09444, 0.02778, 0, 0.5],
    97: [0, 0.44444, 0, 0, 0.48056],
    98: [0, 0.69444, 0, 0, 0.51667],
    99: [0, 0.44444, 0, 0, 0.44445],
    100: [0, 0.69444, 0, 0, 0.51667],
    101: [0, 0.44444, 0, 0, 0.44445],
    102: [0, 0.69444, 0.06944, 0, 0.30556],
    103: [0.19444, 0.44444, 0.01389, 0, 0.5],
    104: [0, 0.69444, 0, 0, 0.51667],
    105: [0, 0.67937, 0, 0, 0.23889],
    106: [0.19444, 0.67937, 0, 0, 0.26667],
    107: [0, 0.69444, 0, 0, 0.48889],
    108: [0, 0.69444, 0, 0, 0.23889],
    109: [0, 0.44444, 0, 0, 0.79445],
    110: [0, 0.44444, 0, 0, 0.51667],
    111: [0, 0.44444, 0, 0, 0.5],
    112: [0.19444, 0.44444, 0, 0, 0.51667],
    113: [0.19444, 0.44444, 0, 0, 0.51667],
    114: [0, 0.44444, 0.01389, 0, 0.34167],
    115: [0, 0.44444, 0, 0, 0.38333],
    116: [0, 0.57143, 0, 0, 0.36111],
    117: [0, 0.44444, 0, 0, 0.51667],
    118: [0, 0.44444, 0.01389, 0, 0.46111],
    119: [0, 0.44444, 0.01389, 0, 0.68334],
    120: [0, 0.44444, 0, 0, 0.46111],
    121: [0.19444, 0.44444, 0.01389, 0, 0.46111],
    122: [0, 0.44444, 0, 0, 0.43472],
    126: [0.35, 0.32659, 0, 0, 0.5],
    160: [0, 0, 0, 0, 0.25],
    168: [0, 0.67937, 0, 0, 0.5],
    176: [0, 0.69444, 0, 0, 0.66667],
    184: [0.17014, 0, 0, 0, 0.44445],
    305: [0, 0.44444, 0, 0, 0.23889],
    567: [0.19444, 0.44444, 0, 0, 0.26667],
    710: [0, 0.69444, 0, 0, 0.5],
    711: [0, 0.63194, 0, 0, 0.5],
    713: [0, 0.60889, 0, 0, 0.5],
    714: [0, 0.69444, 0, 0, 0.5],
    715: [0, 0.69444, 0, 0, 0.5],
    728: [0, 0.69444, 0, 0, 0.5],
    729: [0, 0.67937, 0, 0, 0.27778],
    730: [0, 0.69444, 0, 0, 0.66667],
    732: [0, 0.67659, 0, 0, 0.5],
    733: [0, 0.69444, 0, 0, 0.5],
    915: [0, 0.69444, 0, 0, 0.54167],
    916: [0, 0.69444, 0, 0, 0.83334],
    920: [0, 0.69444, 0, 0, 0.77778],
    923: [0, 0.69444, 0, 0, 0.61111],
    926: [0, 0.69444, 0, 0, 0.66667],
    928: [0, 0.69444, 0, 0, 0.70834],
    931: [0, 0.69444, 0, 0, 0.72222],
    933: [0, 0.69444, 0, 0, 0.77778],
    934: [0, 0.69444, 0, 0, 0.72222],
    936: [0, 0.69444, 0, 0, 0.77778],
    937: [0, 0.69444, 0, 0, 0.72222],
    8211: [0, 0.44444, 0.02778, 0, 0.5],
    8212: [0, 0.44444, 0.02778, 0, 1],
    8216: [0, 0.69444, 0, 0, 0.27778],
    8217: [0, 0.69444, 0, 0, 0.27778],
    8220: [0, 0.69444, 0, 0, 0.5],
    8221: [0, 0.69444, 0, 0, 0.5]
  },
  "Script-Regular": {
    32: [0, 0, 0, 0, 0.25],
    65: [0, 0.7, 0.22925, 0, 0.80253],
    66: [0, 0.7, 0.04087, 0, 0.90757],
    67: [0, 0.7, 0.1689, 0, 0.66619],
    68: [0, 0.7, 0.09371, 0, 0.77443],
    69: [0, 0.7, 0.18583, 0, 0.56162],
    70: [0, 0.7, 0.13634, 0, 0.89544],
    71: [0, 0.7, 0.17322, 0, 0.60961],
    72: [0, 0.7, 0.29694, 0, 0.96919],
    73: [0, 0.7, 0.19189, 0, 0.80907],
    74: [0.27778, 0.7, 0.19189, 0, 1.05159],
    75: [0, 0.7, 0.31259, 0, 0.91364],
    76: [0, 0.7, 0.19189, 0, 0.87373],
    77: [0, 0.7, 0.15981, 0, 1.08031],
    78: [0, 0.7, 0.3525, 0, 0.9015],
    79: [0, 0.7, 0.08078, 0, 0.73787],
    80: [0, 0.7, 0.08078, 0, 1.01262],
    81: [0, 0.7, 0.03305, 0, 0.88282],
    82: [0, 0.7, 0.06259, 0, 0.85],
    83: [0, 0.7, 0.19189, 0, 0.86767],
    84: [0, 0.7, 0.29087, 0, 0.74697],
    85: [0, 0.7, 0.25815, 0, 0.79996],
    86: [0, 0.7, 0.27523, 0, 0.62204],
    87: [0, 0.7, 0.27523, 0, 0.80532],
    88: [0, 0.7, 0.26006, 0, 0.94445],
    89: [0, 0.7, 0.2939, 0, 0.70961],
    90: [0, 0.7, 0.24037, 0, 0.8212],
    160: [0, 0, 0, 0, 0.25]
  },
  "Size1-Regular": {
    32: [0, 0, 0, 0, 0.25],
    40: [0.35001, 0.85, 0, 0, 0.45834],
    41: [0.35001, 0.85, 0, 0, 0.45834],
    47: [0.35001, 0.85, 0, 0, 0.57778],
    91: [0.35001, 0.85, 0, 0, 0.41667],
    92: [0.35001, 0.85, 0, 0, 0.57778],
    93: [0.35001, 0.85, 0, 0, 0.41667],
    123: [0.35001, 0.85, 0, 0, 0.58334],
    125: [0.35001, 0.85, 0, 0, 0.58334],
    160: [0, 0, 0, 0, 0.25],
    710: [0, 0.72222, 0, 0, 0.55556],
    732: [0, 0.72222, 0, 0, 0.55556],
    770: [0, 0.72222, 0, 0, 0.55556],
    771: [0, 0.72222, 0, 0, 0.55556],
    8214: [-99e-5, 0.601, 0, 0, 0.77778],
    8593: [1e-5, 0.6, 0, 0, 0.66667],
    8595: [1e-5, 0.6, 0, 0, 0.66667],
    8657: [1e-5, 0.6, 0, 0, 0.77778],
    8659: [1e-5, 0.6, 0, 0, 0.77778],
    8719: [0.25001, 0.75, 0, 0, 0.94445],
    8720: [0.25001, 0.75, 0, 0, 0.94445],
    8721: [0.25001, 0.75, 0, 0, 1.05556],
    8730: [0.35001, 0.85, 0, 0, 1],
    8739: [-599e-5, 0.606, 0, 0, 0.33333],
    8741: [-599e-5, 0.606, 0, 0, 0.55556],
    8747: [0.30612, 0.805, 0.19445, 0, 0.47222],
    8748: [0.306, 0.805, 0.19445, 0, 0.47222],
    8749: [0.306, 0.805, 0.19445, 0, 0.47222],
    8750: [0.30612, 0.805, 0.19445, 0, 0.47222],
    8896: [0.25001, 0.75, 0, 0, 0.83334],
    8897: [0.25001, 0.75, 0, 0, 0.83334],
    8898: [0.25001, 0.75, 0, 0, 0.83334],
    8899: [0.25001, 0.75, 0, 0, 0.83334],
    8968: [0.35001, 0.85, 0, 0, 0.47222],
    8969: [0.35001, 0.85, 0, 0, 0.47222],
    8970: [0.35001, 0.85, 0, 0, 0.47222],
    8971: [0.35001, 0.85, 0, 0, 0.47222],
    9168: [-99e-5, 0.601, 0, 0, 0.66667],
    10216: [0.35001, 0.85, 0, 0, 0.47222],
    10217: [0.35001, 0.85, 0, 0, 0.47222],
    10752: [0.25001, 0.75, 0, 0, 1.11111],
    10753: [0.25001, 0.75, 0, 0, 1.11111],
    10754: [0.25001, 0.75, 0, 0, 1.11111],
    10756: [0.25001, 0.75, 0, 0, 0.83334],
    10758: [0.25001, 0.75, 0, 0, 0.83334]
  },
  "Size2-Regular": {
    32: [0, 0, 0, 0, 0.25],
    40: [0.65002, 1.15, 0, 0, 0.59722],
    41: [0.65002, 1.15, 0, 0, 0.59722],
    47: [0.65002, 1.15, 0, 0, 0.81111],
    91: [0.65002, 1.15, 0, 0, 0.47222],
    92: [0.65002, 1.15, 0, 0, 0.81111],
    93: [0.65002, 1.15, 0, 0, 0.47222],
    123: [0.65002, 1.15, 0, 0, 0.66667],
    125: [0.65002, 1.15, 0, 0, 0.66667],
    160: [0, 0, 0, 0, 0.25],
    710: [0, 0.75, 0, 0, 1],
    732: [0, 0.75, 0, 0, 1],
    770: [0, 0.75, 0, 0, 1],
    771: [0, 0.75, 0, 0, 1],
    8719: [0.55001, 1.05, 0, 0, 1.27778],
    8720: [0.55001, 1.05, 0, 0, 1.27778],
    8721: [0.55001, 1.05, 0, 0, 1.44445],
    8730: [0.65002, 1.15, 0, 0, 1],
    8747: [0.86225, 1.36, 0.44445, 0, 0.55556],
    8748: [0.862, 1.36, 0.44445, 0, 0.55556],
    8749: [0.862, 1.36, 0.44445, 0, 0.55556],
    8750: [0.86225, 1.36, 0.44445, 0, 0.55556],
    8896: [0.55001, 1.05, 0, 0, 1.11111],
    8897: [0.55001, 1.05, 0, 0, 1.11111],
    8898: [0.55001, 1.05, 0, 0, 1.11111],
    8899: [0.55001, 1.05, 0, 0, 1.11111],
    8968: [0.65002, 1.15, 0, 0, 0.52778],
    8969: [0.65002, 1.15, 0, 0, 0.52778],
    8970: [0.65002, 1.15, 0, 0, 0.52778],
    8971: [0.65002, 1.15, 0, 0, 0.52778],
    10216: [0.65002, 1.15, 0, 0, 0.61111],
    10217: [0.65002, 1.15, 0, 0, 0.61111],
    10752: [0.55001, 1.05, 0, 0, 1.51112],
    10753: [0.55001, 1.05, 0, 0, 1.51112],
    10754: [0.55001, 1.05, 0, 0, 1.51112],
    10756: [0.55001, 1.05, 0, 0, 1.11111],
    10758: [0.55001, 1.05, 0, 0, 1.11111]
  },
  "Size3-Regular": {
    32: [0, 0, 0, 0, 0.25],
    40: [0.95003, 1.45, 0, 0, 0.73611],
    41: [0.95003, 1.45, 0, 0, 0.73611],
    47: [0.95003, 1.45, 0, 0, 1.04445],
    91: [0.95003, 1.45, 0, 0, 0.52778],
    92: [0.95003, 1.45, 0, 0, 1.04445],
    93: [0.95003, 1.45, 0, 0, 0.52778],
    123: [0.95003, 1.45, 0, 0, 0.75],
    125: [0.95003, 1.45, 0, 0, 0.75],
    160: [0, 0, 0, 0, 0.25],
    710: [0, 0.75, 0, 0, 1.44445],
    732: [0, 0.75, 0, 0, 1.44445],
    770: [0, 0.75, 0, 0, 1.44445],
    771: [0, 0.75, 0, 0, 1.44445],
    8730: [0.95003, 1.45, 0, 0, 1],
    8968: [0.95003, 1.45, 0, 0, 0.58334],
    8969: [0.95003, 1.45, 0, 0, 0.58334],
    8970: [0.95003, 1.45, 0, 0, 0.58334],
    8971: [0.95003, 1.45, 0, 0, 0.58334],
    10216: [0.95003, 1.45, 0, 0, 0.75],
    10217: [0.95003, 1.45, 0, 0, 0.75]
  },
  "Size4-Regular": {
    32: [0, 0, 0, 0, 0.25],
    40: [1.25003, 1.75, 0, 0, 0.79167],
    41: [1.25003, 1.75, 0, 0, 0.79167],
    47: [1.25003, 1.75, 0, 0, 1.27778],
    91: [1.25003, 1.75, 0, 0, 0.58334],
    92: [1.25003, 1.75, 0, 0, 1.27778],
    93: [1.25003, 1.75, 0, 0, 0.58334],
    123: [1.25003, 1.75, 0, 0, 0.80556],
    125: [1.25003, 1.75, 0, 0, 0.80556],
    160: [0, 0, 0, 0, 0.25],
    710: [0, 0.825, 0, 0, 1.8889],
    732: [0, 0.825, 0, 0, 1.8889],
    770: [0, 0.825, 0, 0, 1.8889],
    771: [0, 0.825, 0, 0, 1.8889],
    8730: [1.25003, 1.75, 0, 0, 1],
    8968: [1.25003, 1.75, 0, 0, 0.63889],
    8969: [1.25003, 1.75, 0, 0, 0.63889],
    8970: [1.25003, 1.75, 0, 0, 0.63889],
    8971: [1.25003, 1.75, 0, 0, 0.63889],
    9115: [0.64502, 1.155, 0, 0, 0.875],
    9116: [1e-5, 0.6, 0, 0, 0.875],
    9117: [0.64502, 1.155, 0, 0, 0.875],
    9118: [0.64502, 1.155, 0, 0, 0.875],
    9119: [1e-5, 0.6, 0, 0, 0.875],
    9120: [0.64502, 1.155, 0, 0, 0.875],
    9121: [0.64502, 1.155, 0, 0, 0.66667],
    9122: [-99e-5, 0.601, 0, 0, 0.66667],
    9123: [0.64502, 1.155, 0, 0, 0.66667],
    9124: [0.64502, 1.155, 0, 0, 0.66667],
    9125: [-99e-5, 0.601, 0, 0, 0.66667],
    9126: [0.64502, 1.155, 0, 0, 0.66667],
    9127: [1e-5, 0.9, 0, 0, 0.88889],
    9128: [0.65002, 1.15, 0, 0, 0.88889],
    9129: [0.90001, 0, 0, 0, 0.88889],
    9130: [0, 0.3, 0, 0, 0.88889],
    9131: [1e-5, 0.9, 0, 0, 0.88889],
    9132: [0.65002, 1.15, 0, 0, 0.88889],
    9133: [0.90001, 0, 0, 0, 0.88889],
    9143: [0.88502, 0.915, 0, 0, 1.05556],
    10216: [1.25003, 1.75, 0, 0, 0.80556],
    10217: [1.25003, 1.75, 0, 0, 0.80556],
    57344: [-499e-5, 0.605, 0, 0, 1.05556],
    57345: [-499e-5, 0.605, 0, 0, 1.05556],
    57680: [0, 0.12, 0, 0, 0.45],
    57681: [0, 0.12, 0, 0, 0.45],
    57682: [0, 0.12, 0, 0, 0.45],
    57683: [0, 0.12, 0, 0, 0.45]
  },
  "Typewriter-Regular": {
    32: [0, 0, 0, 0, 0.525],
    33: [0, 0.61111, 0, 0, 0.525],
    34: [0, 0.61111, 0, 0, 0.525],
    35: [0, 0.61111, 0, 0, 0.525],
    36: [0.08333, 0.69444, 0, 0, 0.525],
    37: [0.08333, 0.69444, 0, 0, 0.525],
    38: [0, 0.61111, 0, 0, 0.525],
    39: [0, 0.61111, 0, 0, 0.525],
    40: [0.08333, 0.69444, 0, 0, 0.525],
    41: [0.08333, 0.69444, 0, 0, 0.525],
    42: [0, 0.52083, 0, 0, 0.525],
    43: [-0.08056, 0.53055, 0, 0, 0.525],
    44: [0.13889, 0.125, 0, 0, 0.525],
    45: [-0.08056, 0.53055, 0, 0, 0.525],
    46: [0, 0.125, 0, 0, 0.525],
    47: [0.08333, 0.69444, 0, 0, 0.525],
    48: [0, 0.61111, 0, 0, 0.525],
    49: [0, 0.61111, 0, 0, 0.525],
    50: [0, 0.61111, 0, 0, 0.525],
    51: [0, 0.61111, 0, 0, 0.525],
    52: [0, 0.61111, 0, 0, 0.525],
    53: [0, 0.61111, 0, 0, 0.525],
    54: [0, 0.61111, 0, 0, 0.525],
    55: [0, 0.61111, 0, 0, 0.525],
    56: [0, 0.61111, 0, 0, 0.525],
    57: [0, 0.61111, 0, 0, 0.525],
    58: [0, 0.43056, 0, 0, 0.525],
    59: [0.13889, 0.43056, 0, 0, 0.525],
    60: [-0.05556, 0.55556, 0, 0, 0.525],
    61: [-0.19549, 0.41562, 0, 0, 0.525],
    62: [-0.05556, 0.55556, 0, 0, 0.525],
    63: [0, 0.61111, 0, 0, 0.525],
    64: [0, 0.61111, 0, 0, 0.525],
    65: [0, 0.61111, 0, 0, 0.525],
    66: [0, 0.61111, 0, 0, 0.525],
    67: [0, 0.61111, 0, 0, 0.525],
    68: [0, 0.61111, 0, 0, 0.525],
    69: [0, 0.61111, 0, 0, 0.525],
    70: [0, 0.61111, 0, 0, 0.525],
    71: [0, 0.61111, 0, 0, 0.525],
    72: [0, 0.61111, 0, 0, 0.525],
    73: [0, 0.61111, 0, 0, 0.525],
    74: [0, 0.61111, 0, 0, 0.525],
    75: [0, 0.61111, 0, 0, 0.525],
    76: [0, 0.61111, 0, 0, 0.525],
    77: [0, 0.61111, 0, 0, 0.525],
    78: [0, 0.61111, 0, 0, 0.525],
    79: [0, 0.61111, 0, 0, 0.525],
    80: [0, 0.61111, 0, 0, 0.525],
    81: [0.13889, 0.61111, 0, 0, 0.525],
    82: [0, 0.61111, 0, 0, 0.525],
    83: [0, 0.61111, 0, 0, 0.525],
    84: [0, 0.61111, 0, 0, 0.525],
    85: [0, 0.61111, 0, 0, 0.525],
    86: [0, 0.61111, 0, 0, 0.525],
    87: [0, 0.61111, 0, 0, 0.525],
    88: [0, 0.61111, 0, 0, 0.525],
    89: [0, 0.61111, 0, 0, 0.525],
    90: [0, 0.61111, 0, 0, 0.525],
    91: [0.08333, 0.69444, 0, 0, 0.525],
    92: [0.08333, 0.69444, 0, 0, 0.525],
    93: [0.08333, 0.69444, 0, 0, 0.525],
    94: [0, 0.61111, 0, 0, 0.525],
    95: [0.09514, 0, 0, 0, 0.525],
    96: [0, 0.61111, 0, 0, 0.525],
    97: [0, 0.43056, 0, 0, 0.525],
    98: [0, 0.61111, 0, 0, 0.525],
    99: [0, 0.43056, 0, 0, 0.525],
    100: [0, 0.61111, 0, 0, 0.525],
    101: [0, 0.43056, 0, 0, 0.525],
    102: [0, 0.61111, 0, 0, 0.525],
    103: [0.22222, 0.43056, 0, 0, 0.525],
    104: [0, 0.61111, 0, 0, 0.525],
    105: [0, 0.61111, 0, 0, 0.525],
    106: [0.22222, 0.61111, 0, 0, 0.525],
    107: [0, 0.61111, 0, 0, 0.525],
    108: [0, 0.61111, 0, 0, 0.525],
    109: [0, 0.43056, 0, 0, 0.525],
    110: [0, 0.43056, 0, 0, 0.525],
    111: [0, 0.43056, 0, 0, 0.525],
    112: [0.22222, 0.43056, 0, 0, 0.525],
    113: [0.22222, 0.43056, 0, 0, 0.525],
    114: [0, 0.43056, 0, 0, 0.525],
    115: [0, 0.43056, 0, 0, 0.525],
    116: [0, 0.55358, 0, 0, 0.525],
    117: [0, 0.43056, 0, 0, 0.525],
    118: [0, 0.43056, 0, 0, 0.525],
    119: [0, 0.43056, 0, 0, 0.525],
    120: [0, 0.43056, 0, 0, 0.525],
    121: [0.22222, 0.43056, 0, 0, 0.525],
    122: [0, 0.43056, 0, 0, 0.525],
    123: [0.08333, 0.69444, 0, 0, 0.525],
    124: [0.08333, 0.69444, 0, 0, 0.525],
    125: [0.08333, 0.69444, 0, 0, 0.525],
    126: [0, 0.61111, 0, 0, 0.525],
    127: [0, 0.61111, 0, 0, 0.525],
    160: [0, 0, 0, 0, 0.525],
    176: [0, 0.61111, 0, 0, 0.525],
    184: [0.19445, 0, 0, 0, 0.525],
    305: [0, 0.43056, 0, 0, 0.525],
    567: [0.22222, 0.43056, 0, 0, 0.525],
    711: [0, 0.56597, 0, 0, 0.525],
    713: [0, 0.56555, 0, 0, 0.525],
    714: [0, 0.61111, 0, 0, 0.525],
    715: [0, 0.61111, 0, 0, 0.525],
    728: [0, 0.61111, 0, 0, 0.525],
    730: [0, 0.61111, 0, 0, 0.525],
    770: [0, 0.61111, 0, 0, 0.525],
    771: [0, 0.61111, 0, 0, 0.525],
    776: [0, 0.61111, 0, 0, 0.525],
    915: [0, 0.61111, 0, 0, 0.525],
    916: [0, 0.61111, 0, 0, 0.525],
    920: [0, 0.61111, 0, 0, 0.525],
    923: [0, 0.61111, 0, 0, 0.525],
    926: [0, 0.61111, 0, 0, 0.525],
    928: [0, 0.61111, 0, 0, 0.525],
    931: [0, 0.61111, 0, 0, 0.525],
    933: [0, 0.61111, 0, 0, 0.525],
    934: [0, 0.61111, 0, 0, 0.525],
    936: [0, 0.61111, 0, 0, 0.525],
    937: [0, 0.61111, 0, 0, 0.525],
    8216: [0, 0.61111, 0, 0, 0.525],
    8217: [0, 0.61111, 0, 0, 0.525],
    8242: [0, 0.61111, 0, 0, 0.525],
    9251: [0.11111, 0.21944, 0, 0, 0.525]
  }
}, br = {
  slant: [0.25, 0.25, 0.25],
  // sigma1
  space: [0, 0, 0],
  // sigma2
  stretch: [0, 0, 0],
  // sigma3
  shrink: [0, 0, 0],
  // sigma4
  xHeight: [0.431, 0.431, 0.431],
  // sigma5
  quad: [1, 1.171, 1.472],
  // sigma6
  extraSpace: [0, 0, 0],
  // sigma7
  num1: [0.677, 0.732, 0.925],
  // sigma8
  num2: [0.394, 0.384, 0.387],
  // sigma9
  num3: [0.444, 0.471, 0.504],
  // sigma10
  denom1: [0.686, 0.752, 1.025],
  // sigma11
  denom2: [0.345, 0.344, 0.532],
  // sigma12
  sup1: [0.413, 0.503, 0.504],
  // sigma13
  sup2: [0.363, 0.431, 0.404],
  // sigma14
  sup3: [0.289, 0.286, 0.294],
  // sigma15
  sub1: [0.15, 0.143, 0.2],
  // sigma16
  sub2: [0.247, 0.286, 0.4],
  // sigma17
  supDrop: [0.386, 0.353, 0.494],
  // sigma18
  subDrop: [0.05, 0.071, 0.1],
  // sigma19
  delim1: [2.39, 1.7, 1.98],
  // sigma20
  delim2: [1.01, 1.157, 1.42],
  // sigma21
  axisHeight: [0.25, 0.25, 0.25],
  // sigma22
  // These font metrics are extracted from TeX by using tftopl on cmex10.tfm;
  // they correspond to the font parameters of the extension fonts (family 3).
  // See the TeXbook, page 441. In AMSTeX, the extension fonts scale; to
  // match cmex7, we'd use cmex7.tfm values for script and scriptscript
  // values.
  defaultRuleThickness: [0.04, 0.049, 0.049],
  // xi8; cmex7: 0.049
  bigOpSpacing1: [0.111, 0.111, 0.111],
  // xi9
  bigOpSpacing2: [0.166, 0.166, 0.166],
  // xi10
  bigOpSpacing3: [0.2, 0.2, 0.2],
  // xi11
  bigOpSpacing4: [0.6, 0.611, 0.611],
  // xi12; cmex7: 0.611
  bigOpSpacing5: [0.1, 0.143, 0.143],
  // xi13; cmex7: 0.143
  // The \sqrt rule width is taken from the height of the surd character.
  // Since we use the same font at all sizes, this thickness doesn't scale.
  sqrtRuleThickness: [0.04, 0.04, 0.04],
  // This value determines how large a pt is, for metrics which are defined
  // in terms of pts.
  // This value is also used in katex.scss; if you change it make sure the
  // values match.
  ptPerEm: [10, 10, 10],
  // The space between adjacent `|` columns in an array definition. From
  // `\showthe\doublerulesep` in LaTeX. Equals 2.0 / ptPerEm.
  doubleRuleSep: [0.2, 0.2, 0.2],
  // The width of separator lines in {array} environments. From
  // `\showthe\arrayrulewidth` in LaTeX. Equals 0.4 / ptPerEm.
  arrayRuleWidth: [0.04, 0.04, 0.04],
  // Two values from LaTeX source2e:
  fboxsep: [0.3, 0.3, 0.3],
  //        3 pt / ptPerEm
  fboxrule: [0.04, 0.04, 0.04]
  // 0.4 pt / ptPerEm
}, Ai = {
  // Latin-1
  Å: "A",
  Ð: "D",
  Þ: "o",
  å: "a",
  ð: "d",
  þ: "o",
  // Cyrillic
  А: "A",
  Б: "B",
  В: "B",
  Г: "F",
  Д: "A",
  Е: "E",
  Ж: "K",
  З: "3",
  И: "N",
  Й: "N",
  К: "K",
  Л: "N",
  М: "M",
  Н: "H",
  О: "O",
  П: "N",
  Р: "P",
  С: "C",
  Т: "T",
  У: "y",
  Ф: "O",
  Х: "X",
  Ц: "U",
  Ч: "h",
  Ш: "W",
  Щ: "W",
  Ъ: "B",
  Ы: "X",
  Ь: "B",
  Э: "3",
  Ю: "X",
  Я: "R",
  а: "a",
  б: "b",
  в: "a",
  г: "r",
  д: "y",
  е: "e",
  ж: "m",
  з: "e",
  и: "n",
  й: "n",
  к: "n",
  л: "n",
  м: "m",
  н: "n",
  о: "o",
  п: "n",
  р: "p",
  с: "c",
  т: "o",
  у: "y",
  ф: "b",
  х: "x",
  ц: "n",
  ч: "n",
  ш: "w",
  щ: "w",
  ъ: "a",
  ы: "m",
  ь: "a",
  э: "e",
  ю: "m",
  я: "r"
};
function As(n, e) {
  zt[n] = e;
}
function Sa(n, e, t) {
  if (!zt[e])
    throw new Error("Font metrics not found for font: " + e + ".");
  var r = n.charCodeAt(0), a = zt[e][r];
  if (!a && n[0] in Ai && (r = Ai[n[0]].charCodeAt(0), a = zt[e][r]), !a && t === "text" && Ss(r) && (a = zt[e][77]), a)
    return {
      depth: a[0],
      height: a[1],
      italic: a[2],
      skew: a[3],
      width: a[4]
    };
}
var Dn = {};
function v1(n) {
  var e;
  if (n >= 5 ? e = 0 : n >= 3 ? e = 1 : e = 2, !Dn[e]) {
    var t = Dn[e] = {
      cssEmPerMu: br.quad[e] / 18
    };
    for (var r in br)
      br.hasOwnProperty(r) && (t[r] = br[r][e]);
  }
  return Dn[e];
}
var _1 = [
  // Each element contains [textsize, scriptsize, scriptscriptsize].
  // The size mappings are taken from TeX with \normalsize=10pt.
  [1, 1, 1],
  // size1: [5, 5, 5]              \tiny
  [2, 1, 1],
  // size2: [6, 5, 5]
  [3, 1, 1],
  // size3: [7, 5, 5]              \scriptsize
  [4, 2, 1],
  // size4: [8, 6, 5]              \footnotesize
  [5, 2, 1],
  // size5: [9, 6, 5]              \small
  [6, 3, 1],
  // size6: [10, 7, 5]             \normalsize
  [7, 4, 2],
  // size7: [12, 8, 6]             \large
  [8, 6, 3],
  // size8: [14.4, 10, 7]          \Large
  [9, 7, 6],
  // size9: [17.28, 12, 10]        \LARGE
  [10, 8, 7],
  // size10: [20.74, 14.4, 12]     \huge
  [11, 10, 9]
  // size11: [24.88, 20.74, 17.28] \HUGE
], Ei = [
  // fontMetrics.js:getGlobalMetrics also uses size indexes, so if
  // you change size indexes, change that function.
  0.5,
  0.6,
  0.7,
  0.8,
  0.9,
  1,
  1.2,
  1.44,
  1.728,
  2.074,
  2.488
], Fi = function(e, t) {
  return t.size < 2 ? e : _1[e - 1][t.size - 1];
};
class Jt {
  // A font family applies to a group of fonts (i.e. SansSerif), while a font
  // represents a specific font (i.e. SansSerif Bold).
  // See: https://tex.stackexchange.com/questions/22350/difference-between-textrm-and-mathrm
  /**
   * The base size index.
   */
  constructor(e) {
    this.style = void 0, this.color = void 0, this.size = void 0, this.textSize = void 0, this.phantom = void 0, this.font = void 0, this.fontFamily = void 0, this.fontWeight = void 0, this.fontShape = void 0, this.sizeMultiplier = void 0, this.maxSize = void 0, this.minRuleThickness = void 0, this._fontMetrics = void 0, this.style = e.style, this.color = e.color, this.size = e.size || Jt.BASESIZE, this.textSize = e.textSize || this.size, this.phantom = !!e.phantom, this.font = e.font || "", this.fontFamily = e.fontFamily || "", this.fontWeight = e.fontWeight || "", this.fontShape = e.fontShape || "", this.sizeMultiplier = Ei[this.size - 1], this.maxSize = e.maxSize, this.minRuleThickness = e.minRuleThickness, this._fontMetrics = void 0;
  }
  /**
   * Returns a new options object with the same properties as "this".  Properties
   * from "extension" will be copied to the new options object.
   */
  extend(e) {
    var t = {
      style: this.style,
      size: this.size,
      textSize: this.textSize,
      color: this.color,
      phantom: this.phantom,
      font: this.font,
      fontFamily: this.fontFamily,
      fontWeight: this.fontWeight,
      fontShape: this.fontShape,
      maxSize: this.maxSize,
      minRuleThickness: this.minRuleThickness
    };
    for (var r in e)
      e.hasOwnProperty(r) && (t[r] = e[r]);
    return new Jt(t);
  }
  /**
   * Return an options object with the given style. If `this.style === style`,
   * returns `this`.
   */
  havingStyle(e) {
    return this.style === e ? this : this.extend({
      style: e,
      size: Fi(this.textSize, e)
    });
  }
  /**
   * Return an options object with a cramped version of the current style. If
   * the current style is cramped, returns `this`.
   */
  havingCrampedStyle() {
    return this.havingStyle(this.style.cramp());
  }
  /**
   * Return an options object with the given size and in at least `\textstyle`.
   * Returns `this` if appropriate.
   */
  havingSize(e) {
    return this.size === e && this.textSize === e ? this : this.extend({
      style: this.style.text(),
      size: e,
      textSize: e,
      sizeMultiplier: Ei[e - 1]
    });
  }
  /**
   * Like `this.havingSize(BASESIZE).havingStyle(style)`. If `style` is omitted,
   * changes to at least `\textstyle`.
   */
  havingBaseStyle(e) {
    e = e || this.style.text();
    var t = Fi(Jt.BASESIZE, e);
    return this.size === t && this.textSize === Jt.BASESIZE && this.style === e ? this : this.extend({
      style: e,
      size: t
    });
  }
  /**
   * Remove the effect of sizing changes such as \Huge.
   * Keep the effect of the current style, such as \scriptstyle.
   */
  havingBaseSizing() {
    var e;
    switch (this.style.id) {
      case 4:
      case 5:
        e = 3;
        break;
      case 6:
      case 7:
        e = 1;
        break;
      default:
        e = 6;
    }
    return this.extend({
      style: this.style.text(),
      size: e
    });
  }
  /**
   * Create a new options object with the given color.
   */
  withColor(e) {
    return this.extend({
      color: e
    });
  }
  /**
   * Create a new options object with "phantom" set to true.
   */
  withPhantom() {
    return this.extend({
      phantom: !0
    });
  }
  /**
   * Creates a new options object with the given math font or old text font.
   * @type {[type]}
   */
  withFont(e) {
    return this.extend({
      font: e
    });
  }
  /**
   * Create a new options objects with the given fontFamily.
   */
  withTextFontFamily(e) {
    return this.extend({
      fontFamily: e,
      font: ""
    });
  }
  /**
   * Creates a new options object with the given font weight
   */
  withTextFontWeight(e) {
    return this.extend({
      fontWeight: e,
      font: ""
    });
  }
  /**
   * Creates a new options object with the given font weight
   */
  withTextFontShape(e) {
    return this.extend({
      fontShape: e,
      font: ""
    });
  }
  /**
   * Return the CSS sizing classes required to switch from enclosing options
   * `oldOptions` to `this`. Returns an array of classes.
   */
  sizingClasses(e) {
    return e.size !== this.size ? ["sizing", "reset-size" + e.size, "size" + this.size] : [];
  }
  /**
   * Return the CSS sizing classes required to switch to the base size. Like
   * `this.havingSize(BASESIZE).sizingClasses(this)`.
   */
  baseSizingClasses() {
    return this.size !== Jt.BASESIZE ? ["sizing", "reset-size" + this.size, "size" + Jt.BASESIZE] : [];
  }
  /**
   * Return the font metrics for this size.
   */
  fontMetrics() {
    return this._fontMetrics || (this._fontMetrics = v1(this.size)), this._fontMetrics;
  }
  /**
   * Gets the CSS color of the current options object
   */
  getColor() {
    return this.phantom ? "transparent" : this.color;
  }
}
Jt.BASESIZE = 6;
var ta = {
  // https://en.wikibooks.org/wiki/LaTeX/Lengths and
  // https://tex.stackexchange.com/a/8263
  pt: 1,
  // TeX point
  mm: 7227 / 2540,
  // millimeter
  cm: 7227 / 254,
  // centimeter
  in: 72.27,
  // inch
  bp: 803 / 800,
  // big (PostScript) points
  pc: 12,
  // pica
  dd: 1238 / 1157,
  // didot
  cc: 14856 / 1157,
  // cicero (12 didot)
  nd: 685 / 642,
  // new didot
  nc: 1370 / 107,
  // new cicero (12 new didot)
  sp: 1 / 65536,
  // scaled point (TeX's internal smallest unit)
  // https://tex.stackexchange.com/a/41371
  px: 803 / 800
  // \pdfpxdimen defaults to 1 bp in pdfTeX and LuaTeX
}, b1 = {
  ex: !0,
  em: !0,
  mu: !0
}, Es = function(e) {
  return typeof e != "string" && (e = e.unit), e in ta || e in b1 || e === "ex";
}, ke = function(e, t) {
  var r;
  if (e.unit in ta)
    r = ta[e.unit] / t.fontMetrics().ptPerEm / t.sizeMultiplier;
  else if (e.unit === "mu")
    r = t.fontMetrics().cssEmPerMu;
  else {
    var a;
    if (t.style.isTight() ? a = t.havingStyle(t.style.text()) : a = t, e.unit === "ex")
      r = a.fontMetrics().xHeight;
    else if (e.unit === "em")
      r = a.fontMetrics().quad;
    else
      throw new L("Invalid unit: '" + e.unit + "'");
    a !== t && (r *= a.sizeMultiplier / t.sizeMultiplier);
  }
  return Math.min(e.number * r, t.maxSize);
}, P = function(e) {
  return +e.toFixed(4) + "em";
}, d0 = function(e) {
  return e.filter((t) => t).join(" ");
}, Fs = function(e, t, r) {
  if (this.classes = e || [], this.attributes = {}, this.height = 0, this.depth = 0, this.maxFontSize = 0, this.style = r || {}, t) {
    t.style.isTight() && this.classes.push("mtight");
    var a = t.getColor();
    a && (this.style.color = a);
  }
}, Cs = function(e) {
  var t = document.createElement(e);
  t.className = d0(this.classes);
  for (var r in this.style)
    this.style.hasOwnProperty(r) && (t.style[r] = this.style[r]);
  for (var a in this.attributes)
    this.attributes.hasOwnProperty(a) && t.setAttribute(a, this.attributes[a]);
  for (var i = 0; i < this.children.length; i++)
    t.appendChild(this.children[i].toNode());
  return t;
}, y1 = /[\s"'>/=\x00-\x1f]/, Ts = function(e) {
  var t = "<" + e;
  this.classes.length && (t += ' class="' + Z.escape(d0(this.classes)) + '"');
  var r = "";
  for (var a in this.style)
    this.style.hasOwnProperty(a) && (r += Z.hyphenate(a) + ":" + this.style[a] + ";");
  r && (t += ' style="' + Z.escape(r) + '"');
  for (var i in this.attributes)
    if (this.attributes.hasOwnProperty(i)) {
      if (y1.test(i))
        throw new L("Invalid attribute name '" + i + "'");
      t += " " + i + '="' + Z.escape(this.attributes[i]) + '"';
    }
  t += ">";
  for (var l = 0; l < this.children.length; l++)
    t += this.children[l].toMarkup();
  return t += "</" + e + ">", t;
};
class cr {
  constructor(e, t, r, a) {
    this.children = void 0, this.attributes = void 0, this.classes = void 0, this.height = void 0, this.depth = void 0, this.width = void 0, this.maxFontSize = void 0, this.style = void 0, Fs.call(this, e, r, a), this.children = t || [];
  }
  /**
   * Sets an arbitrary attribute on the span. Warning: use this wisely. Not
   * all browsers support attributes the same, and having too many custom
   * attributes is probably bad.
   */
  setAttribute(e, t) {
    this.attributes[e] = t;
  }
  hasClass(e) {
    return Z.contains(this.classes, e);
  }
  toNode() {
    return Cs.call(this, "span");
  }
  toMarkup() {
    return Ts.call(this, "span");
  }
}
class Aa {
  constructor(e, t, r, a) {
    this.children = void 0, this.attributes = void 0, this.classes = void 0, this.height = void 0, this.depth = void 0, this.maxFontSize = void 0, this.style = void 0, Fs.call(this, t, a), this.children = r || [], this.setAttribute("href", e);
  }
  setAttribute(e, t) {
    this.attributes[e] = t;
  }
  hasClass(e) {
    return Z.contains(this.classes, e);
  }
  toNode() {
    return Cs.call(this, "a");
  }
  toMarkup() {
    return Ts.call(this, "a");
  }
}
class w1 {
  constructor(e, t, r) {
    this.src = void 0, this.alt = void 0, this.classes = void 0, this.height = void 0, this.depth = void 0, this.maxFontSize = void 0, this.style = void 0, this.alt = t, this.src = e, this.classes = ["mord"], this.style = r;
  }
  hasClass(e) {
    return Z.contains(this.classes, e);
  }
  toNode() {
    var e = document.createElement("img");
    e.src = this.src, e.alt = this.alt, e.className = "mord";
    for (var t in this.style)
      this.style.hasOwnProperty(t) && (e.style[t] = this.style[t]);
    return e;
  }
  toMarkup() {
    var e = '<img src="' + Z.escape(this.src) + '"' + (' alt="' + Z.escape(this.alt) + '"'), t = "";
    for (var r in this.style)
      this.style.hasOwnProperty(r) && (t += Z.hyphenate(r) + ":" + this.style[r] + ";");
    return t && (e += ' style="' + Z.escape(t) + '"'), e += "'/>", e;
  }
}
var x1 = {
  î: "ı̂",
  ï: "ı̈",
  í: "ı́",
  // 'ī': '\u0131\u0304', // enable when we add Extended Latin
  ì: "ı̀"
};
class ut {
  constructor(e, t, r, a, i, l, s, u) {
    this.text = void 0, this.height = void 0, this.depth = void 0, this.italic = void 0, this.skew = void 0, this.width = void 0, this.maxFontSize = void 0, this.classes = void 0, this.style = void 0, this.text = e, this.height = t || 0, this.depth = r || 0, this.italic = a || 0, this.skew = i || 0, this.width = l || 0, this.classes = s || [], this.style = u || {}, this.maxFontSize = 0;
    var h = l1(this.text.charCodeAt(0));
    h && this.classes.push(h + "_fallback"), /[îïíì]/.test(this.text) && (this.text = x1[this.text]);
  }
  hasClass(e) {
    return Z.contains(this.classes, e);
  }
  /**
   * Creates a text node or span from a symbol node. Note that a span is only
   * created if it is needed.
   */
  toNode() {
    var e = document.createTextNode(this.text), t = null;
    this.italic > 0 && (t = document.createElement("span"), t.style.marginRight = P(this.italic)), this.classes.length > 0 && (t = t || document.createElement("span"), t.className = d0(this.classes));
    for (var r in this.style)
      this.style.hasOwnProperty(r) && (t = t || document.createElement("span"), t.style[r] = this.style[r]);
    return t ? (t.appendChild(e), t) : e;
  }
  /**
   * Creates markup for a symbol node.
   */
  toMarkup() {
    var e = !1, t = "<span";
    this.classes.length && (e = !0, t += ' class="', t += Z.escape(d0(this.classes)), t += '"');
    var r = "";
    this.italic > 0 && (r += "margin-right:" + this.italic + "em;");
    for (var a in this.style)
      this.style.hasOwnProperty(a) && (r += Z.hyphenate(a) + ":" + this.style[a] + ";");
    r && (e = !0, t += ' style="' + Z.escape(r) + '"');
    var i = Z.escape(this.text);
    return e ? (t += ">", t += i, t += "</span>", t) : i;
  }
}
class r0 {
  constructor(e, t) {
    this.children = void 0, this.attributes = void 0, this.children = e || [], this.attributes = t || {};
  }
  toNode() {
    var e = "http://www.w3.org/2000/svg", t = document.createElementNS(e, "svg");
    for (var r in this.attributes)
      Object.prototype.hasOwnProperty.call(this.attributes, r) && t.setAttribute(r, this.attributes[r]);
    for (var a = 0; a < this.children.length; a++)
      t.appendChild(this.children[a].toNode());
    return t;
  }
  toMarkup() {
    var e = '<svg xmlns="http://www.w3.org/2000/svg"';
    for (var t in this.attributes)
      Object.prototype.hasOwnProperty.call(this.attributes, t) && (e += " " + t + '="' + Z.escape(this.attributes[t]) + '"');
    e += ">";
    for (var r = 0; r < this.children.length; r++)
      e += this.children[r].toMarkup();
    return e += "</svg>", e;
  }
}
class m0 {
  constructor(e, t) {
    this.pathName = void 0, this.alternate = void 0, this.pathName = e, this.alternate = t;
  }
  toNode() {
    var e = "http://www.w3.org/2000/svg", t = document.createElementNS(e, "path");
    return this.alternate ? t.setAttribute("d", this.alternate) : t.setAttribute("d", Si[this.pathName]), t;
  }
  toMarkup() {
    return this.alternate ? '<path d="' + Z.escape(this.alternate) + '"/>' : '<path d="' + Z.escape(Si[this.pathName]) + '"/>';
  }
}
class ra {
  constructor(e) {
    this.attributes = void 0, this.attributes = e || {};
  }
  toNode() {
    var e = "http://www.w3.org/2000/svg", t = document.createElementNS(e, "line");
    for (var r in this.attributes)
      Object.prototype.hasOwnProperty.call(this.attributes, r) && t.setAttribute(r, this.attributes[r]);
    return t;
  }
  toMarkup() {
    var e = "<line";
    for (var t in this.attributes)
      Object.prototype.hasOwnProperty.call(this.attributes, t) && (e += " " + t + '="' + Z.escape(this.attributes[t]) + '"');
    return e += "/>", e;
  }
}
function Ci(n) {
  if (n instanceof ut)
    return n;
  throw new Error("Expected symbolNode but got " + String(n) + ".");
}
function k1(n) {
  if (n instanceof cr)
    return n;
  throw new Error("Expected span<HtmlDomNode> but got " + String(n) + ".");
}
var D1 = {
  bin: 1,
  close: 1,
  inner: 1,
  open: 1,
  punct: 1,
  rel: 1
}, S1 = {
  "accent-token": 1,
  mathord: 1,
  "op-token": 1,
  spacing: 1,
  textord: 1
}, ge = {
  math: {},
  text: {}
};
function o(n, e, t, r, a, i) {
  ge[n][a] = {
    font: e,
    group: t,
    replace: r
  }, i && r && (ge[n][r] = ge[n][a]);
}
var c = "math", R = "text", m = "main", b = "ams", ye = "accent-token", V = "bin", We = "close", I0 = "inner", K = "mathord", Te = "op-token", rt = "open", ln = "punct", y = "rel", i0 = "spacing", D = "textord";
o(c, m, y, "≡", "\\equiv", !0);
o(c, m, y, "≺", "\\prec", !0);
o(c, m, y, "≻", "\\succ", !0);
o(c, m, y, "∼", "\\sim", !0);
o(c, m, y, "⊥", "\\perp");
o(c, m, y, "⪯", "\\preceq", !0);
o(c, m, y, "⪰", "\\succeq", !0);
o(c, m, y, "≃", "\\simeq", !0);
o(c, m, y, "∣", "\\mid", !0);
o(c, m, y, "≪", "\\ll", !0);
o(c, m, y, "≫", "\\gg", !0);
o(c, m, y, "≍", "\\asymp", !0);
o(c, m, y, "∥", "\\parallel");
o(c, m, y, "⋈", "\\bowtie", !0);
o(c, m, y, "⌣", "\\smile", !0);
o(c, m, y, "⊑", "\\sqsubseteq", !0);
o(c, m, y, "⊒", "\\sqsupseteq", !0);
o(c, m, y, "≐", "\\doteq", !0);
o(c, m, y, "⌢", "\\frown", !0);
o(c, m, y, "∋", "\\ni", !0);
o(c, m, y, "∝", "\\propto", !0);
o(c, m, y, "⊢", "\\vdash", !0);
o(c, m, y, "⊣", "\\dashv", !0);
o(c, m, y, "∋", "\\owns");
o(c, m, ln, ".", "\\ldotp");
o(c, m, ln, "⋅", "\\cdotp");
o(c, m, D, "#", "\\#");
o(R, m, D, "#", "\\#");
o(c, m, D, "&", "\\&");
o(R, m, D, "&", "\\&");
o(c, m, D, "ℵ", "\\aleph", !0);
o(c, m, D, "∀", "\\forall", !0);
o(c, m, D, "ℏ", "\\hbar", !0);
o(c, m, D, "∃", "\\exists", !0);
o(c, m, D, "∇", "\\nabla", !0);
o(c, m, D, "♭", "\\flat", !0);
o(c, m, D, "ℓ", "\\ell", !0);
o(c, m, D, "♮", "\\natural", !0);
o(c, m, D, "♣", "\\clubsuit", !0);
o(c, m, D, "℘", "\\wp", !0);
o(c, m, D, "♯", "\\sharp", !0);
o(c, m, D, "♢", "\\diamondsuit", !0);
o(c, m, D, "ℜ", "\\Re", !0);
o(c, m, D, "♡", "\\heartsuit", !0);
o(c, m, D, "ℑ", "\\Im", !0);
o(c, m, D, "♠", "\\spadesuit", !0);
o(c, m, D, "§", "\\S", !0);
o(R, m, D, "§", "\\S");
o(c, m, D, "¶", "\\P", !0);
o(R, m, D, "¶", "\\P");
o(c, m, D, "†", "\\dag");
o(R, m, D, "†", "\\dag");
o(R, m, D, "†", "\\textdagger");
o(c, m, D, "‡", "\\ddag");
o(R, m, D, "‡", "\\ddag");
o(R, m, D, "‡", "\\textdaggerdbl");
o(c, m, We, "⎱", "\\rmoustache", !0);
o(c, m, rt, "⎰", "\\lmoustache", !0);
o(c, m, We, "⟯", "\\rgroup", !0);
o(c, m, rt, "⟮", "\\lgroup", !0);
o(c, m, V, "∓", "\\mp", !0);
o(c, m, V, "⊖", "\\ominus", !0);
o(c, m, V, "⊎", "\\uplus", !0);
o(c, m, V, "⊓", "\\sqcap", !0);
o(c, m, V, "∗", "\\ast");
o(c, m, V, "⊔", "\\sqcup", !0);
o(c, m, V, "◯", "\\bigcirc", !0);
o(c, m, V, "∙", "\\bullet", !0);
o(c, m, V, "‡", "\\ddagger");
o(c, m, V, "≀", "\\wr", !0);
o(c, m, V, "⨿", "\\amalg");
o(c, m, V, "&", "\\And");
o(c, m, y, "⟵", "\\longleftarrow", !0);
o(c, m, y, "⇐", "\\Leftarrow", !0);
o(c, m, y, "⟸", "\\Longleftarrow", !0);
o(c, m, y, "⟶", "\\longrightarrow", !0);
o(c, m, y, "⇒", "\\Rightarrow", !0);
o(c, m, y, "⟹", "\\Longrightarrow", !0);
o(c, m, y, "↔", "\\leftrightarrow", !0);
o(c, m, y, "⟷", "\\longleftrightarrow", !0);
o(c, m, y, "⇔", "\\Leftrightarrow", !0);
o(c, m, y, "⟺", "\\Longleftrightarrow", !0);
o(c, m, y, "↦", "\\mapsto", !0);
o(c, m, y, "⟼", "\\longmapsto", !0);
o(c, m, y, "↗", "\\nearrow", !0);
o(c, m, y, "↩", "\\hookleftarrow", !0);
o(c, m, y, "↪", "\\hookrightarrow", !0);
o(c, m, y, "↘", "\\searrow", !0);
o(c, m, y, "↼", "\\leftharpoonup", !0);
o(c, m, y, "⇀", "\\rightharpoonup", !0);
o(c, m, y, "↙", "\\swarrow", !0);
o(c, m, y, "↽", "\\leftharpoondown", !0);
o(c, m, y, "⇁", "\\rightharpoondown", !0);
o(c, m, y, "↖", "\\nwarrow", !0);
o(c, m, y, "⇌", "\\rightleftharpoons", !0);
o(c, b, y, "≮", "\\nless", !0);
o(c, b, y, "", "\\@nleqslant");
o(c, b, y, "", "\\@nleqq");
o(c, b, y, "⪇", "\\lneq", !0);
o(c, b, y, "≨", "\\lneqq", !0);
o(c, b, y, "", "\\@lvertneqq");
o(c, b, y, "⋦", "\\lnsim", !0);
o(c, b, y, "⪉", "\\lnapprox", !0);
o(c, b, y, "⊀", "\\nprec", !0);
o(c, b, y, "⋠", "\\npreceq", !0);
o(c, b, y, "⋨", "\\precnsim", !0);
o(c, b, y, "⪹", "\\precnapprox", !0);
o(c, b, y, "≁", "\\nsim", !0);
o(c, b, y, "", "\\@nshortmid");
o(c, b, y, "∤", "\\nmid", !0);
o(c, b, y, "⊬", "\\nvdash", !0);
o(c, b, y, "⊭", "\\nvDash", !0);
o(c, b, y, "⋪", "\\ntriangleleft");
o(c, b, y, "⋬", "\\ntrianglelefteq", !0);
o(c, b, y, "⊊", "\\subsetneq", !0);
o(c, b, y, "", "\\@varsubsetneq");
o(c, b, y, "⫋", "\\subsetneqq", !0);
o(c, b, y, "", "\\@varsubsetneqq");
o(c, b, y, "≯", "\\ngtr", !0);
o(c, b, y, "", "\\@ngeqslant");
o(c, b, y, "", "\\@ngeqq");
o(c, b, y, "⪈", "\\gneq", !0);
o(c, b, y, "≩", "\\gneqq", !0);
o(c, b, y, "", "\\@gvertneqq");
o(c, b, y, "⋧", "\\gnsim", !0);
o(c, b, y, "⪊", "\\gnapprox", !0);
o(c, b, y, "⊁", "\\nsucc", !0);
o(c, b, y, "⋡", "\\nsucceq", !0);
o(c, b, y, "⋩", "\\succnsim", !0);
o(c, b, y, "⪺", "\\succnapprox", !0);
o(c, b, y, "≆", "\\ncong", !0);
o(c, b, y, "", "\\@nshortparallel");
o(c, b, y, "∦", "\\nparallel", !0);
o(c, b, y, "⊯", "\\nVDash", !0);
o(c, b, y, "⋫", "\\ntriangleright");
o(c, b, y, "⋭", "\\ntrianglerighteq", !0);
o(c, b, y, "", "\\@nsupseteqq");
o(c, b, y, "⊋", "\\supsetneq", !0);
o(c, b, y, "", "\\@varsupsetneq");
o(c, b, y, "⫌", "\\supsetneqq", !0);
o(c, b, y, "", "\\@varsupsetneqq");
o(c, b, y, "⊮", "\\nVdash", !0);
o(c, b, y, "⪵", "\\precneqq", !0);
o(c, b, y, "⪶", "\\succneqq", !0);
o(c, b, y, "", "\\@nsubseteqq");
o(c, b, V, "⊴", "\\unlhd");
o(c, b, V, "⊵", "\\unrhd");
o(c, b, y, "↚", "\\nleftarrow", !0);
o(c, b, y, "↛", "\\nrightarrow", !0);
o(c, b, y, "⇍", "\\nLeftarrow", !0);
o(c, b, y, "⇏", "\\nRightarrow", !0);
o(c, b, y, "↮", "\\nleftrightarrow", !0);
o(c, b, y, "⇎", "\\nLeftrightarrow", !0);
o(c, b, y, "△", "\\vartriangle");
o(c, b, D, "ℏ", "\\hslash");
o(c, b, D, "▽", "\\triangledown");
o(c, b, D, "◊", "\\lozenge");
o(c, b, D, "Ⓢ", "\\circledS");
o(c, b, D, "®", "\\circledR");
o(R, b, D, "®", "\\circledR");
o(c, b, D, "∡", "\\measuredangle", !0);
o(c, b, D, "∄", "\\nexists");
o(c, b, D, "℧", "\\mho");
o(c, b, D, "Ⅎ", "\\Finv", !0);
o(c, b, D, "⅁", "\\Game", !0);
o(c, b, D, "‵", "\\backprime");
o(c, b, D, "▲", "\\blacktriangle");
o(c, b, D, "▼", "\\blacktriangledown");
o(c, b, D, "■", "\\blacksquare");
o(c, b, D, "⧫", "\\blacklozenge");
o(c, b, D, "★", "\\bigstar");
o(c, b, D, "∢", "\\sphericalangle", !0);
o(c, b, D, "∁", "\\complement", !0);
o(c, b, D, "ð", "\\eth", !0);
o(R, m, D, "ð", "ð");
o(c, b, D, "╱", "\\diagup");
o(c, b, D, "╲", "\\diagdown");
o(c, b, D, "□", "\\square");
o(c, b, D, "□", "\\Box");
o(c, b, D, "◊", "\\Diamond");
o(c, b, D, "¥", "\\yen", !0);
o(R, b, D, "¥", "\\yen", !0);
o(c, b, D, "✓", "\\checkmark", !0);
o(R, b, D, "✓", "\\checkmark");
o(c, b, D, "ℶ", "\\beth", !0);
o(c, b, D, "ℸ", "\\daleth", !0);
o(c, b, D, "ℷ", "\\gimel", !0);
o(c, b, D, "ϝ", "\\digamma", !0);
o(c, b, D, "ϰ", "\\varkappa");
o(c, b, rt, "┌", "\\@ulcorner", !0);
o(c, b, We, "┐", "\\@urcorner", !0);
o(c, b, rt, "└", "\\@llcorner", !0);
o(c, b, We, "┘", "\\@lrcorner", !0);
o(c, b, y, "≦", "\\leqq", !0);
o(c, b, y, "⩽", "\\leqslant", !0);
o(c, b, y, "⪕", "\\eqslantless", !0);
o(c, b, y, "≲", "\\lesssim", !0);
o(c, b, y, "⪅", "\\lessapprox", !0);
o(c, b, y, "≊", "\\approxeq", !0);
o(c, b, V, "⋖", "\\lessdot");
o(c, b, y, "⋘", "\\lll", !0);
o(c, b, y, "≶", "\\lessgtr", !0);
o(c, b, y, "⋚", "\\lesseqgtr", !0);
o(c, b, y, "⪋", "\\lesseqqgtr", !0);
o(c, b, y, "≑", "\\doteqdot");
o(c, b, y, "≓", "\\risingdotseq", !0);
o(c, b, y, "≒", "\\fallingdotseq", !0);
o(c, b, y, "∽", "\\backsim", !0);
o(c, b, y, "⋍", "\\backsimeq", !0);
o(c, b, y, "⫅", "\\subseteqq", !0);
o(c, b, y, "⋐", "\\Subset", !0);
o(c, b, y, "⊏", "\\sqsubset", !0);
o(c, b, y, "≼", "\\preccurlyeq", !0);
o(c, b, y, "⋞", "\\curlyeqprec", !0);
o(c, b, y, "≾", "\\precsim", !0);
o(c, b, y, "⪷", "\\precapprox", !0);
o(c, b, y, "⊲", "\\vartriangleleft");
o(c, b, y, "⊴", "\\trianglelefteq");
o(c, b, y, "⊨", "\\vDash", !0);
o(c, b, y, "⊪", "\\Vvdash", !0);
o(c, b, y, "⌣", "\\smallsmile");
o(c, b, y, "⌢", "\\smallfrown");
o(c, b, y, "≏", "\\bumpeq", !0);
o(c, b, y, "≎", "\\Bumpeq", !0);
o(c, b, y, "≧", "\\geqq", !0);
o(c, b, y, "⩾", "\\geqslant", !0);
o(c, b, y, "⪖", "\\eqslantgtr", !0);
o(c, b, y, "≳", "\\gtrsim", !0);
o(c, b, y, "⪆", "\\gtrapprox", !0);
o(c, b, V, "⋗", "\\gtrdot");
o(c, b, y, "⋙", "\\ggg", !0);
o(c, b, y, "≷", "\\gtrless", !0);
o(c, b, y, "⋛", "\\gtreqless", !0);
o(c, b, y, "⪌", "\\gtreqqless", !0);
o(c, b, y, "≖", "\\eqcirc", !0);
o(c, b, y, "≗", "\\circeq", !0);
o(c, b, y, "≜", "\\triangleq", !0);
o(c, b, y, "∼", "\\thicksim");
o(c, b, y, "≈", "\\thickapprox");
o(c, b, y, "⫆", "\\supseteqq", !0);
o(c, b, y, "⋑", "\\Supset", !0);
o(c, b, y, "⊐", "\\sqsupset", !0);
o(c, b, y, "≽", "\\succcurlyeq", !0);
o(c, b, y, "⋟", "\\curlyeqsucc", !0);
o(c, b, y, "≿", "\\succsim", !0);
o(c, b, y, "⪸", "\\succapprox", !0);
o(c, b, y, "⊳", "\\vartriangleright");
o(c, b, y, "⊵", "\\trianglerighteq");
o(c, b, y, "⊩", "\\Vdash", !0);
o(c, b, y, "∣", "\\shortmid");
o(c, b, y, "∥", "\\shortparallel");
o(c, b, y, "≬", "\\between", !0);
o(c, b, y, "⋔", "\\pitchfork", !0);
o(c, b, y, "∝", "\\varpropto");
o(c, b, y, "◀", "\\blacktriangleleft");
o(c, b, y, "∴", "\\therefore", !0);
o(c, b, y, "∍", "\\backepsilon");
o(c, b, y, "▶", "\\blacktriangleright");
o(c, b, y, "∵", "\\because", !0);
o(c, b, y, "⋘", "\\llless");
o(c, b, y, "⋙", "\\gggtr");
o(c, b, V, "⊲", "\\lhd");
o(c, b, V, "⊳", "\\rhd");
o(c, b, y, "≂", "\\eqsim", !0);
o(c, m, y, "⋈", "\\Join");
o(c, b, y, "≑", "\\Doteq", !0);
o(c, b, V, "∔", "\\dotplus", !0);
o(c, b, V, "∖", "\\smallsetminus");
o(c, b, V, "⋒", "\\Cap", !0);
o(c, b, V, "⋓", "\\Cup", !0);
o(c, b, V, "⩞", "\\doublebarwedge", !0);
o(c, b, V, "⊟", "\\boxminus", !0);
o(c, b, V, "⊞", "\\boxplus", !0);
o(c, b, V, "⋇", "\\divideontimes", !0);
o(c, b, V, "⋉", "\\ltimes", !0);
o(c, b, V, "⋊", "\\rtimes", !0);
o(c, b, V, "⋋", "\\leftthreetimes", !0);
o(c, b, V, "⋌", "\\rightthreetimes", !0);
o(c, b, V, "⋏", "\\curlywedge", !0);
o(c, b, V, "⋎", "\\curlyvee", !0);
o(c, b, V, "⊝", "\\circleddash", !0);
o(c, b, V, "⊛", "\\circledast", !0);
o(c, b, V, "⋅", "\\centerdot");
o(c, b, V, "⊺", "\\intercal", !0);
o(c, b, V, "⋒", "\\doublecap");
o(c, b, V, "⋓", "\\doublecup");
o(c, b, V, "⊠", "\\boxtimes", !0);
o(c, b, y, "⇢", "\\dashrightarrow", !0);
o(c, b, y, "⇠", "\\dashleftarrow", !0);
o(c, b, y, "⇇", "\\leftleftarrows", !0);
o(c, b, y, "⇆", "\\leftrightarrows", !0);
o(c, b, y, "⇚", "\\Lleftarrow", !0);
o(c, b, y, "↞", "\\twoheadleftarrow", !0);
o(c, b, y, "↢", "\\leftarrowtail", !0);
o(c, b, y, "↫", "\\looparrowleft", !0);
o(c, b, y, "⇋", "\\leftrightharpoons", !0);
o(c, b, y, "↶", "\\curvearrowleft", !0);
o(c, b, y, "↺", "\\circlearrowleft", !0);
o(c, b, y, "↰", "\\Lsh", !0);
o(c, b, y, "⇈", "\\upuparrows", !0);
o(c, b, y, "↿", "\\upharpoonleft", !0);
o(c, b, y, "⇃", "\\downharpoonleft", !0);
o(c, m, y, "⊶", "\\origof", !0);
o(c, m, y, "⊷", "\\imageof", !0);
o(c, b, y, "⊸", "\\multimap", !0);
o(c, b, y, "↭", "\\leftrightsquigarrow", !0);
o(c, b, y, "⇉", "\\rightrightarrows", !0);
o(c, b, y, "⇄", "\\rightleftarrows", !0);
o(c, b, y, "↠", "\\twoheadrightarrow", !0);
o(c, b, y, "↣", "\\rightarrowtail", !0);
o(c, b, y, "↬", "\\looparrowright", !0);
o(c, b, y, "↷", "\\curvearrowright", !0);
o(c, b, y, "↻", "\\circlearrowright", !0);
o(c, b, y, "↱", "\\Rsh", !0);
o(c, b, y, "⇊", "\\downdownarrows", !0);
o(c, b, y, "↾", "\\upharpoonright", !0);
o(c, b, y, "⇂", "\\downharpoonright", !0);
o(c, b, y, "⇝", "\\rightsquigarrow", !0);
o(c, b, y, "⇝", "\\leadsto");
o(c, b, y, "⇛", "\\Rrightarrow", !0);
o(c, b, y, "↾", "\\restriction");
o(c, m, D, "‘", "`");
o(c, m, D, "$", "\\$");
o(R, m, D, "$", "\\$");
o(R, m, D, "$", "\\textdollar");
o(c, m, D, "%", "\\%");
o(R, m, D, "%", "\\%");
o(c, m, D, "_", "\\_");
o(R, m, D, "_", "\\_");
o(R, m, D, "_", "\\textunderscore");
o(c, m, D, "∠", "\\angle", !0);
o(c, m, D, "∞", "\\infty", !0);
o(c, m, D, "′", "\\prime");
o(c, m, D, "△", "\\triangle");
o(c, m, D, "Γ", "\\Gamma", !0);
o(c, m, D, "Δ", "\\Delta", !0);
o(c, m, D, "Θ", "\\Theta", !0);
o(c, m, D, "Λ", "\\Lambda", !0);
o(c, m, D, "Ξ", "\\Xi", !0);
o(c, m, D, "Π", "\\Pi", !0);
o(c, m, D, "Σ", "\\Sigma", !0);
o(c, m, D, "Υ", "\\Upsilon", !0);
o(c, m, D, "Φ", "\\Phi", !0);
o(c, m, D, "Ψ", "\\Psi", !0);
o(c, m, D, "Ω", "\\Omega", !0);
o(c, m, D, "A", "Α");
o(c, m, D, "B", "Β");
o(c, m, D, "E", "Ε");
o(c, m, D, "Z", "Ζ");
o(c, m, D, "H", "Η");
o(c, m, D, "I", "Ι");
o(c, m, D, "K", "Κ");
o(c, m, D, "M", "Μ");
o(c, m, D, "N", "Ν");
o(c, m, D, "O", "Ο");
o(c, m, D, "P", "Ρ");
o(c, m, D, "T", "Τ");
o(c, m, D, "X", "Χ");
o(c, m, D, "¬", "\\neg", !0);
o(c, m, D, "¬", "\\lnot");
o(c, m, D, "⊤", "\\top");
o(c, m, D, "⊥", "\\bot");
o(c, m, D, "∅", "\\emptyset");
o(c, b, D, "∅", "\\varnothing");
o(c, m, K, "α", "\\alpha", !0);
o(c, m, K, "β", "\\beta", !0);
o(c, m, K, "γ", "\\gamma", !0);
o(c, m, K, "δ", "\\delta", !0);
o(c, m, K, "ϵ", "\\epsilon", !0);
o(c, m, K, "ζ", "\\zeta", !0);
o(c, m, K, "η", "\\eta", !0);
o(c, m, K, "θ", "\\theta", !0);
o(c, m, K, "ι", "\\iota", !0);
o(c, m, K, "κ", "\\kappa", !0);
o(c, m, K, "λ", "\\lambda", !0);
o(c, m, K, "μ", "\\mu", !0);
o(c, m, K, "ν", "\\nu", !0);
o(c, m, K, "ξ", "\\xi", !0);
o(c, m, K, "ο", "\\omicron", !0);
o(c, m, K, "π", "\\pi", !0);
o(c, m, K, "ρ", "\\rho", !0);
o(c, m, K, "σ", "\\sigma", !0);
o(c, m, K, "τ", "\\tau", !0);
o(c, m, K, "υ", "\\upsilon", !0);
o(c, m, K, "ϕ", "\\phi", !0);
o(c, m, K, "χ", "\\chi", !0);
o(c, m, K, "ψ", "\\psi", !0);
o(c, m, K, "ω", "\\omega", !0);
o(c, m, K, "ε", "\\varepsilon", !0);
o(c, m, K, "ϑ", "\\vartheta", !0);
o(c, m, K, "ϖ", "\\varpi", !0);
o(c, m, K, "ϱ", "\\varrho", !0);
o(c, m, K, "ς", "\\varsigma", !0);
o(c, m, K, "φ", "\\varphi", !0);
o(c, m, V, "∗", "*", !0);
o(c, m, V, "+", "+");
o(c, m, V, "−", "-", !0);
o(c, m, V, "⋅", "\\cdot", !0);
o(c, m, V, "∘", "\\circ", !0);
o(c, m, V, "÷", "\\div", !0);
o(c, m, V, "±", "\\pm", !0);
o(c, m, V, "×", "\\times", !0);
o(c, m, V, "∩", "\\cap", !0);
o(c, m, V, "∪", "\\cup", !0);
o(c, m, V, "∖", "\\setminus", !0);
o(c, m, V, "∧", "\\land");
o(c, m, V, "∨", "\\lor");
o(c, m, V, "∧", "\\wedge", !0);
o(c, m, V, "∨", "\\vee", !0);
o(c, m, D, "√", "\\surd");
o(c, m, rt, "⟨", "\\langle", !0);
o(c, m, rt, "∣", "\\lvert");
o(c, m, rt, "∥", "\\lVert");
o(c, m, We, "?", "?");
o(c, m, We, "!", "!");
o(c, m, We, "⟩", "\\rangle", !0);
o(c, m, We, "∣", "\\rvert");
o(c, m, We, "∥", "\\rVert");
o(c, m, y, "=", "=");
o(c, m, y, ":", ":");
o(c, m, y, "≈", "\\approx", !0);
o(c, m, y, "≅", "\\cong", !0);
o(c, m, y, "≥", "\\ge");
o(c, m, y, "≥", "\\geq", !0);
o(c, m, y, "←", "\\gets");
o(c, m, y, ">", "\\gt", !0);
o(c, m, y, "∈", "\\in", !0);
o(c, m, y, "", "\\@not");
o(c, m, y, "⊂", "\\subset", !0);
o(c, m, y, "⊃", "\\supset", !0);
o(c, m, y, "⊆", "\\subseteq", !0);
o(c, m, y, "⊇", "\\supseteq", !0);
o(c, b, y, "⊈", "\\nsubseteq", !0);
o(c, b, y, "⊉", "\\nsupseteq", !0);
o(c, m, y, "⊨", "\\models");
o(c, m, y, "←", "\\leftarrow", !0);
o(c, m, y, "≤", "\\le");
o(c, m, y, "≤", "\\leq", !0);
o(c, m, y, "<", "\\lt", !0);
o(c, m, y, "→", "\\rightarrow", !0);
o(c, m, y, "→", "\\to");
o(c, b, y, "≱", "\\ngeq", !0);
o(c, b, y, "≰", "\\nleq", !0);
o(c, m, i0, " ", "\\ ");
o(c, m, i0, " ", "\\space");
o(c, m, i0, " ", "\\nobreakspace");
o(R, m, i0, " ", "\\ ");
o(R, m, i0, " ", " ");
o(R, m, i0, " ", "\\space");
o(R, m, i0, " ", "\\nobreakspace");
o(c, m, i0, null, "\\nobreak");
o(c, m, i0, null, "\\allowbreak");
o(c, m, ln, ",", ",");
o(c, m, ln, ";", ";");
o(c, b, V, "⊼", "\\barwedge", !0);
o(c, b, V, "⊻", "\\veebar", !0);
o(c, m, V, "⊙", "\\odot", !0);
o(c, m, V, "⊕", "\\oplus", !0);
o(c, m, V, "⊗", "\\otimes", !0);
o(c, m, D, "∂", "\\partial", !0);
o(c, m, V, "⊘", "\\oslash", !0);
o(c, b, V, "⊚", "\\circledcirc", !0);
o(c, b, V, "⊡", "\\boxdot", !0);
o(c, m, V, "△", "\\bigtriangleup");
o(c, m, V, "▽", "\\bigtriangledown");
o(c, m, V, "†", "\\dagger");
o(c, m, V, "⋄", "\\diamond");
o(c, m, V, "⋆", "\\star");
o(c, m, V, "◃", "\\triangleleft");
o(c, m, V, "▹", "\\triangleright");
o(c, m, rt, "{", "\\{");
o(R, m, D, "{", "\\{");
o(R, m, D, "{", "\\textbraceleft");
o(c, m, We, "}", "\\}");
o(R, m, D, "}", "\\}");
o(R, m, D, "}", "\\textbraceright");
o(c, m, rt, "{", "\\lbrace");
o(c, m, We, "}", "\\rbrace");
o(c, m, rt, "[", "\\lbrack", !0);
o(R, m, D, "[", "\\lbrack", !0);
o(c, m, We, "]", "\\rbrack", !0);
o(R, m, D, "]", "\\rbrack", !0);
o(c, m, rt, "(", "\\lparen", !0);
o(c, m, We, ")", "\\rparen", !0);
o(R, m, D, "<", "\\textless", !0);
o(R, m, D, ">", "\\textgreater", !0);
o(c, m, rt, "⌊", "\\lfloor", !0);
o(c, m, We, "⌋", "\\rfloor", !0);
o(c, m, rt, "⌈", "\\lceil", !0);
o(c, m, We, "⌉", "\\rceil", !0);
o(c, m, D, "\\", "\\backslash");
o(c, m, D, "∣", "|");
o(c, m, D, "∣", "\\vert");
o(R, m, D, "|", "\\textbar", !0);
o(c, m, D, "∥", "\\|");
o(c, m, D, "∥", "\\Vert");
o(R, m, D, "∥", "\\textbardbl");
o(R, m, D, "~", "\\textasciitilde");
o(R, m, D, "\\", "\\textbackslash");
o(R, m, D, "^", "\\textasciicircum");
o(c, m, y, "↑", "\\uparrow", !0);
o(c, m, y, "⇑", "\\Uparrow", !0);
o(c, m, y, "↓", "\\downarrow", !0);
o(c, m, y, "⇓", "\\Downarrow", !0);
o(c, m, y, "↕", "\\updownarrow", !0);
o(c, m, y, "⇕", "\\Updownarrow", !0);
o(c, m, Te, "∐", "\\coprod");
o(c, m, Te, "⋁", "\\bigvee");
o(c, m, Te, "⋀", "\\bigwedge");
o(c, m, Te, "⨄", "\\biguplus");
o(c, m, Te, "⋂", "\\bigcap");
o(c, m, Te, "⋃", "\\bigcup");
o(c, m, Te, "∫", "\\int");
o(c, m, Te, "∫", "\\intop");
o(c, m, Te, "∬", "\\iint");
o(c, m, Te, "∭", "\\iiint");
o(c, m, Te, "∏", "\\prod");
o(c, m, Te, "∑", "\\sum");
o(c, m, Te, "⨂", "\\bigotimes");
o(c, m, Te, "⨁", "\\bigoplus");
o(c, m, Te, "⨀", "\\bigodot");
o(c, m, Te, "∮", "\\oint");
o(c, m, Te, "∯", "\\oiint");
o(c, m, Te, "∰", "\\oiiint");
o(c, m, Te, "⨆", "\\bigsqcup");
o(c, m, Te, "∫", "\\smallint");
o(R, m, I0, "…", "\\textellipsis");
o(c, m, I0, "…", "\\mathellipsis");
o(R, m, I0, "…", "\\ldots", !0);
o(c, m, I0, "…", "\\ldots", !0);
o(c, m, I0, "⋯", "\\@cdots", !0);
o(c, m, I0, "⋱", "\\ddots", !0);
o(c, m, D, "⋮", "\\varvdots");
o(R, m, D, "⋮", "\\varvdots");
o(c, m, ye, "ˊ", "\\acute");
o(c, m, ye, "ˋ", "\\grave");
o(c, m, ye, "¨", "\\ddot");
o(c, m, ye, "~", "\\tilde");
o(c, m, ye, "ˉ", "\\bar");
o(c, m, ye, "˘", "\\breve");
o(c, m, ye, "ˇ", "\\check");
o(c, m, ye, "^", "\\hat");
o(c, m, ye, "⃗", "\\vec");
o(c, m, ye, "˙", "\\dot");
o(c, m, ye, "˚", "\\mathring");
o(c, m, K, "", "\\@imath");
o(c, m, K, "", "\\@jmath");
o(c, m, D, "ı", "ı");
o(c, m, D, "ȷ", "ȷ");
o(R, m, D, "ı", "\\i", !0);
o(R, m, D, "ȷ", "\\j", !0);
o(R, m, D, "ß", "\\ss", !0);
o(R, m, D, "æ", "\\ae", !0);
o(R, m, D, "œ", "\\oe", !0);
o(R, m, D, "ø", "\\o", !0);
o(R, m, D, "Æ", "\\AE", !0);
o(R, m, D, "Œ", "\\OE", !0);
o(R, m, D, "Ø", "\\O", !0);
o(R, m, ye, "ˊ", "\\'");
o(R, m, ye, "ˋ", "\\`");
o(R, m, ye, "ˆ", "\\^");
o(R, m, ye, "˜", "\\~");
o(R, m, ye, "ˉ", "\\=");
o(R, m, ye, "˘", "\\u");
o(R, m, ye, "˙", "\\.");
o(R, m, ye, "¸", "\\c");
o(R, m, ye, "˚", "\\r");
o(R, m, ye, "ˇ", "\\v");
o(R, m, ye, "¨", '\\"');
o(R, m, ye, "˝", "\\H");
o(R, m, ye, "◯", "\\textcircled");
var $s = {
  "--": !0,
  "---": !0,
  "``": !0,
  "''": !0
};
o(R, m, D, "–", "--", !0);
o(R, m, D, "–", "\\textendash");
o(R, m, D, "—", "---", !0);
o(R, m, D, "—", "\\textemdash");
o(R, m, D, "‘", "`", !0);
o(R, m, D, "‘", "\\textquoteleft");
o(R, m, D, "’", "'", !0);
o(R, m, D, "’", "\\textquoteright");
o(R, m, D, "“", "``", !0);
o(R, m, D, "“", "\\textquotedblleft");
o(R, m, D, "”", "''", !0);
o(R, m, D, "”", "\\textquotedblright");
o(c, m, D, "°", "\\degree", !0);
o(R, m, D, "°", "\\degree");
o(R, m, D, "°", "\\textdegree", !0);
o(c, m, D, "£", "\\pounds");
o(c, m, D, "£", "\\mathsterling", !0);
o(R, m, D, "£", "\\pounds");
o(R, m, D, "£", "\\textsterling", !0);
o(c, b, D, "✠", "\\maltese");
o(R, b, D, "✠", "\\maltese");
var Ti = '0123456789/@."';
for (var Sn = 0; Sn < Ti.length; Sn++) {
  var $i = Ti.charAt(Sn);
  o(c, m, D, $i, $i);
}
var Mi = '0123456789!@*()-=+";:?/.,';
for (var An = 0; An < Mi.length; An++) {
  var zi = Mi.charAt(An);
  o(R, m, D, zi, zi);
}
var Kr = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
for (var En = 0; En < Kr.length; En++) {
  var yr = Kr.charAt(En);
  o(c, m, K, yr, yr), o(R, m, D, yr, yr);
}
o(c, b, D, "C", "ℂ");
o(R, b, D, "C", "ℂ");
o(c, b, D, "H", "ℍ");
o(R, b, D, "H", "ℍ");
o(c, b, D, "N", "ℕ");
o(R, b, D, "N", "ℕ");
o(c, b, D, "P", "ℙ");
o(R, b, D, "P", "ℙ");
o(c, b, D, "Q", "ℚ");
o(R, b, D, "Q", "ℚ");
o(c, b, D, "R", "ℝ");
o(R, b, D, "R", "ℝ");
o(c, b, D, "Z", "ℤ");
o(R, b, D, "Z", "ℤ");
o(c, m, K, "h", "ℎ");
o(R, m, K, "h", "ℎ");
var J = "";
for (var Pe = 0; Pe < Kr.length; Pe++) {
  var Ae = Kr.charAt(Pe);
  J = String.fromCharCode(55349, 56320 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56372 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56424 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56580 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56684 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56736 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56788 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56840 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56944 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), Pe < 26 && (J = String.fromCharCode(55349, 56632 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J), J = String.fromCharCode(55349, 56476 + Pe), o(c, m, K, Ae, J), o(R, m, D, Ae, J));
}
J = "𝕜";
o(c, m, K, "k", J);
o(R, m, D, "k", J);
for (var g0 = 0; g0 < 10; g0++) {
  var s0 = g0.toString();
  J = String.fromCharCode(55349, 57294 + g0), o(c, m, K, s0, J), o(R, m, D, s0, J), J = String.fromCharCode(55349, 57314 + g0), o(c, m, K, s0, J), o(R, m, D, s0, J), J = String.fromCharCode(55349, 57324 + g0), o(c, m, K, s0, J), o(R, m, D, s0, J), J = String.fromCharCode(55349, 57334 + g0), o(c, m, K, s0, J), o(R, m, D, s0, J);
}
var na = "ÐÞþ";
for (var Fn = 0; Fn < na.length; Fn++) {
  var wr = na.charAt(Fn);
  o(c, m, K, wr, wr), o(R, m, D, wr, wr);
}
var xr = [
  ["mathbf", "textbf", "Main-Bold"],
  // A-Z bold upright
  ["mathbf", "textbf", "Main-Bold"],
  // a-z bold upright
  ["mathnormal", "textit", "Math-Italic"],
  // A-Z italic
  ["mathnormal", "textit", "Math-Italic"],
  // a-z italic
  ["boldsymbol", "boldsymbol", "Main-BoldItalic"],
  // A-Z bold italic
  ["boldsymbol", "boldsymbol", "Main-BoldItalic"],
  // a-z bold italic
  // Map fancy A-Z letters to script, not calligraphic.
  // This aligns with unicode-math and math fonts (except Cambria Math).
  ["mathscr", "textscr", "Script-Regular"],
  // A-Z script
  ["", "", ""],
  // a-z script.  No font
  ["", "", ""],
  // A-Z bold script. No font
  ["", "", ""],
  // a-z bold script. No font
  ["mathfrak", "textfrak", "Fraktur-Regular"],
  // A-Z Fraktur
  ["mathfrak", "textfrak", "Fraktur-Regular"],
  // a-z Fraktur
  ["mathbb", "textbb", "AMS-Regular"],
  // A-Z double-struck
  ["mathbb", "textbb", "AMS-Regular"],
  // k double-struck
  // Note that we are using a bold font, but font metrics for regular Fraktur.
  ["mathboldfrak", "textboldfrak", "Fraktur-Regular"],
  // A-Z bold Fraktur
  ["mathboldfrak", "textboldfrak", "Fraktur-Regular"],
  // a-z bold Fraktur
  ["mathsf", "textsf", "SansSerif-Regular"],
  // A-Z sans-serif
  ["mathsf", "textsf", "SansSerif-Regular"],
  // a-z sans-serif
  ["mathboldsf", "textboldsf", "SansSerif-Bold"],
  // A-Z bold sans-serif
  ["mathboldsf", "textboldsf", "SansSerif-Bold"],
  // a-z bold sans-serif
  ["mathitsf", "textitsf", "SansSerif-Italic"],
  // A-Z italic sans-serif
  ["mathitsf", "textitsf", "SansSerif-Italic"],
  // a-z italic sans-serif
  ["", "", ""],
  // A-Z bold italic sans. No font
  ["", "", ""],
  // a-z bold italic sans. No font
  ["mathtt", "texttt", "Typewriter-Regular"],
  // A-Z monospace
  ["mathtt", "texttt", "Typewriter-Regular"]
  // a-z monospace
], Bi = [
  ["mathbf", "textbf", "Main-Bold"],
  // 0-9 bold
  ["", "", ""],
  // 0-9 double-struck. No KaTeX font.
  ["mathsf", "textsf", "SansSerif-Regular"],
  // 0-9 sans-serif
  ["mathboldsf", "textboldsf", "SansSerif-Bold"],
  // 0-9 bold sans-serif
  ["mathtt", "texttt", "Typewriter-Regular"]
  // 0-9 monospace
], A1 = function(e, t) {
  var r = e.charCodeAt(0), a = e.charCodeAt(1), i = (r - 55296) * 1024 + (a - 56320) + 65536, l = t === "math" ? 0 : 1;
  if (119808 <= i && i < 120484) {
    var s = Math.floor((i - 119808) / 26);
    return [xr[s][2], xr[s][l]];
  } else if (120782 <= i && i <= 120831) {
    var u = Math.floor((i - 120782) / 10);
    return [Bi[u][2], Bi[u][l]];
  } else {
    if (i === 120485 || i === 120486)
      return [xr[0][2], xr[0][l]];
    if (120486 < i && i < 120782)
      return ["", ""];
    throw new L("Unsupported character: " + e);
  }
}, sn = function(e, t, r) {
  return ge[r][e] && ge[r][e].replace && (e = ge[r][e].replace), {
    value: e,
    metrics: Sa(e, t, r)
  };
}, xt = function(e, t, r, a, i) {
  var l = sn(e, t, r), s = l.metrics;
  e = l.value;
  var u;
  if (s) {
    var h = s.italic;
    (r === "text" || a && a.font === "mathit") && (h = 0), u = new ut(e, s.height, s.depth, h, s.skew, s.width, i);
  } else
    typeof console < "u" && console.warn("No character metrics " + ("for '" + e + "' in style '" + t + "' and mode '" + r + "'")), u = new ut(e, 0, 0, 0, 0, 0, i);
  if (a) {
    u.maxFontSize = a.sizeMultiplier, a.style.isTight() && u.classes.push("mtight");
    var d = a.getColor();
    d && (u.style.color = d);
  }
  return u;
}, E1 = function(e, t, r, a) {
  return a === void 0 && (a = []), r.font === "boldsymbol" && sn(e, "Main-Bold", t).metrics ? xt(e, "Main-Bold", t, r, a.concat(["mathbf"])) : e === "\\" || ge[t][e].font === "main" ? xt(e, "Main-Regular", t, r, a) : xt(e, "AMS-Regular", t, r, a.concat(["amsrm"]));
}, F1 = function(e, t, r, a, i) {
  return i !== "textord" && sn(e, "Math-BoldItalic", t).metrics ? {
    fontName: "Math-BoldItalic",
    fontClass: "boldsymbol"
  } : {
    fontName: "Main-Bold",
    fontClass: "mathbf"
  };
}, C1 = function(e, t, r) {
  var a = e.mode, i = e.text, l = ["mord"], s = a === "math" || a === "text" && t.font, u = s ? t.font : t.fontFamily, h = "", d = "";
  if (i.charCodeAt(0) === 55349 && ([h, d] = A1(i, a)), h.length > 0)
    return xt(i, h, a, t, l.concat(d));
  if (u) {
    var g, p;
    if (u === "boldsymbol") {
      var v = F1(i, a, t, l, r);
      g = v.fontName, p = [v.fontClass];
    } else s ? (g = Bs[u].fontName, p = [u]) : (g = kr(u, t.fontWeight, t.fontShape), p = [u, t.fontWeight, t.fontShape]);
    if (sn(i, g, a).metrics)
      return xt(i, g, a, t, l.concat(p));
    if ($s.hasOwnProperty(i) && g.slice(0, 10) === "Typewriter") {
      for (var k = [], A = 0; A < i.length; A++)
        k.push(xt(i[A], g, a, t, l.concat(p)));
      return zs(k);
    }
  }
  if (r === "mathord")
    return xt(i, "Math-Italic", a, t, l.concat(["mathnormal"]));
  if (r === "textord") {
    var C = ge[a][i] && ge[a][i].font;
    if (C === "ams") {
      var z = kr("amsrm", t.fontWeight, t.fontShape);
      return xt(i, z, a, t, l.concat("amsrm", t.fontWeight, t.fontShape));
    } else if (C === "main" || !C) {
      var x = kr("textrm", t.fontWeight, t.fontShape);
      return xt(i, x, a, t, l.concat(t.fontWeight, t.fontShape));
    } else {
      var _ = kr(C, t.fontWeight, t.fontShape);
      return xt(i, _, a, t, l.concat(_, t.fontWeight, t.fontShape));
    }
  } else
    throw new Error("unexpected type: " + r + " in makeOrd");
}, T1 = (n, e) => {
  if (d0(n.classes) !== d0(e.classes) || n.skew !== e.skew || n.maxFontSize !== e.maxFontSize)
    return !1;
  if (n.classes.length === 1) {
    var t = n.classes[0];
    if (t === "mbin" || t === "mord")
      return !1;
  }
  for (var r in n.style)
    if (n.style.hasOwnProperty(r) && n.style[r] !== e.style[r])
      return !1;
  for (var a in e.style)
    if (e.style.hasOwnProperty(a) && n.style[a] !== e.style[a])
      return !1;
  return !0;
}, $1 = (n) => {
  for (var e = 0; e < n.length - 1; e++) {
    var t = n[e], r = n[e + 1];
    t instanceof ut && r instanceof ut && T1(t, r) && (t.text += r.text, t.height = Math.max(t.height, r.height), t.depth = Math.max(t.depth, r.depth), t.italic = r.italic, n.splice(e + 1, 1), e--);
  }
  return n;
}, Ea = function(e) {
  for (var t = 0, r = 0, a = 0, i = 0; i < e.children.length; i++) {
    var l = e.children[i];
    l.height > t && (t = l.height), l.depth > r && (r = l.depth), l.maxFontSize > a && (a = l.maxFontSize);
  }
  e.height = t, e.depth = r, e.maxFontSize = a;
}, Xe = function(e, t, r, a) {
  var i = new cr(e, t, r, a);
  return Ea(i), i;
}, Ms = (n, e, t, r) => new cr(n, e, t, r), M1 = function(e, t, r) {
  var a = Xe([e], [], t);
  return a.height = Math.max(r || t.fontMetrics().defaultRuleThickness, t.minRuleThickness), a.style.borderBottomWidth = P(a.height), a.maxFontSize = 1, a;
}, z1 = function(e, t, r, a) {
  var i = new Aa(e, t, r, a);
  return Ea(i), i;
}, zs = function(e) {
  var t = new ur(e);
  return Ea(t), t;
}, B1 = function(e, t) {
  return e instanceof ur ? Xe([], [e], t) : e;
}, R1 = function(e) {
  if (e.positionType === "individualShift") {
    for (var t = e.children, r = [t[0]], a = -t[0].shift - t[0].elem.depth, i = a, l = 1; l < t.length; l++) {
      var s = -t[l].shift - i - t[l].elem.depth, u = s - (t[l - 1].elem.height + t[l - 1].elem.depth);
      i = i + s, r.push({
        type: "kern",
        size: u
      }), r.push(t[l]);
    }
    return {
      children: r,
      depth: a
    };
  }
  var h;
  if (e.positionType === "top") {
    for (var d = e.positionData, g = 0; g < e.children.length; g++) {
      var p = e.children[g];
      d -= p.type === "kern" ? p.size : p.elem.height + p.elem.depth;
    }
    h = d;
  } else if (e.positionType === "bottom")
    h = -e.positionData;
  else {
    var v = e.children[0];
    if (v.type !== "elem")
      throw new Error('First child must have type "elem".');
    if (e.positionType === "shift")
      h = -v.elem.depth - e.positionData;
    else if (e.positionType === "firstBaseline")
      h = -v.elem.depth;
    else
      throw new Error("Invalid positionType " + e.positionType + ".");
  }
  return {
    children: e.children,
    depth: h
  };
}, N1 = function(e, t) {
  for (var {
    children: r,
    depth: a
  } = R1(e), i = 0, l = 0; l < r.length; l++) {
    var s = r[l];
    if (s.type === "elem") {
      var u = s.elem;
      i = Math.max(i, u.maxFontSize, u.height);
    }
  }
  i += 2;
  var h = Xe(["pstrut"], []);
  h.style.height = P(i);
  for (var d = [], g = a, p = a, v = a, k = 0; k < r.length; k++) {
    var A = r[k];
    if (A.type === "kern")
      v += A.size;
    else {
      var C = A.elem, z = A.wrapperClasses || [], x = A.wrapperStyle || {}, _ = Xe(z, [h, C], void 0, x);
      _.style.top = P(-i - v - C.depth), A.marginLeft && (_.style.marginLeft = A.marginLeft), A.marginRight && (_.style.marginRight = A.marginRight), d.push(_), v += C.height + C.depth;
    }
    g = Math.min(g, v), p = Math.max(p, v);
  }
  var w = Xe(["vlist"], d);
  w.style.height = P(p);
  var E;
  if (g < 0) {
    var T = Xe([], []), $ = Xe(["vlist"], [T]);
    $.style.height = P(-g);
    var M = Xe(["vlist-s"], [new ut("​")]);
    E = [Xe(["vlist-r"], [w, M]), Xe(["vlist-r"], [$])];
  } else
    E = [Xe(["vlist-r"], [w])];
  var B = Xe(["vlist-t"], E);
  return E.length === 2 && B.classes.push("vlist-t2"), B.height = p, B.depth = -g, B;
}, q1 = (n, e) => {
  var t = Xe(["mspace"], [], e), r = ke(n, e);
  return t.style.marginRight = P(r), t;
}, kr = function(e, t, r) {
  var a = "";
  switch (e) {
    case "amsrm":
      a = "AMS";
      break;
    case "textrm":
      a = "Main";
      break;
    case "textsf":
      a = "SansSerif";
      break;
    case "texttt":
      a = "Typewriter";
      break;
    default:
      a = e;
  }
  var i;
  return t === "textbf" && r === "textit" ? i = "BoldItalic" : t === "textbf" ? i = "Bold" : t === "textit" ? i = "Italic" : i = "Regular", a + "-" + i;
}, Bs = {
  // styles
  mathbf: {
    variant: "bold",
    fontName: "Main-Bold"
  },
  mathrm: {
    variant: "normal",
    fontName: "Main-Regular"
  },
  textit: {
    variant: "italic",
    fontName: "Main-Italic"
  },
  mathit: {
    variant: "italic",
    fontName: "Main-Italic"
  },
  mathnormal: {
    variant: "italic",
    fontName: "Math-Italic"
  },
  mathsfit: {
    variant: "sans-serif-italic",
    fontName: "SansSerif-Italic"
  },
  // "boldsymbol" is missing because they require the use of multiple fonts:
  // Math-BoldItalic and Main-Bold.  This is handled by a special case in
  // makeOrd which ends up calling boldsymbol.
  // families
  mathbb: {
    variant: "double-struck",
    fontName: "AMS-Regular"
  },
  mathcal: {
    variant: "script",
    fontName: "Caligraphic-Regular"
  },
  mathfrak: {
    variant: "fraktur",
    fontName: "Fraktur-Regular"
  },
  mathscr: {
    variant: "script",
    fontName: "Script-Regular"
  },
  mathsf: {
    variant: "sans-serif",
    fontName: "SansSerif-Regular"
  },
  mathtt: {
    variant: "monospace",
    fontName: "Typewriter-Regular"
  }
}, Rs = {
  //   path, width, height
  vec: ["vec", 0.471, 0.714],
  // values from the font glyph
  oiintSize1: ["oiintSize1", 0.957, 0.499],
  // oval to overlay the integrand
  oiintSize2: ["oiintSize2", 1.472, 0.659],
  oiiintSize1: ["oiiintSize1", 1.304, 0.499],
  oiiintSize2: ["oiiintSize2", 1.98, 0.659]
}, L1 = function(e, t) {
  var [r, a, i] = Rs[e], l = new m0(r), s = new r0([l], {
    width: P(a),
    height: P(i),
    // Override CSS rule `.katex svg { width: 100% }`
    style: "width:" + P(a),
    viewBox: "0 0 " + 1e3 * a + " " + 1e3 * i,
    preserveAspectRatio: "xMinYMin"
  }), u = Ms(["overlay"], [s], t);
  return u.height = i, u.style.height = P(i), u.style.width = P(a), u;
}, F = {
  fontMap: Bs,
  makeSymbol: xt,
  mathsym: E1,
  makeSpan: Xe,
  makeSvgSpan: Ms,
  makeLineSpan: M1,
  makeAnchor: z1,
  makeFragment: zs,
  wrapFragment: B1,
  makeVList: N1,
  makeOrd: C1,
  makeGlue: q1,
  staticSvg: L1,
  svgData: Rs,
  tryCombineChars: $1
}, xe = {
  number: 3,
  unit: "mu"
}, v0 = {
  number: 4,
  unit: "mu"
}, Zt = {
  number: 5,
  unit: "mu"
}, I1 = {
  mord: {
    mop: xe,
    mbin: v0,
    mrel: Zt,
    minner: xe
  },
  mop: {
    mord: xe,
    mop: xe,
    mrel: Zt,
    minner: xe
  },
  mbin: {
    mord: v0,
    mop: v0,
    mopen: v0,
    minner: v0
  },
  mrel: {
    mord: Zt,
    mop: Zt,
    mopen: Zt,
    minner: Zt
  },
  mopen: {},
  mclose: {
    mop: xe,
    mbin: v0,
    mrel: Zt,
    minner: xe
  },
  mpunct: {
    mord: xe,
    mop: xe,
    mrel: Zt,
    mopen: xe,
    mclose: xe,
    mpunct: xe,
    minner: xe
  },
  minner: {
    mord: xe,
    mop: xe,
    mbin: v0,
    mrel: Zt,
    mopen: xe,
    mpunct: xe,
    minner: xe
  }
}, O1 = {
  mord: {
    mop: xe
  },
  mop: {
    mord: xe,
    mop: xe
  },
  mbin: {},
  mrel: {},
  mopen: {},
  mclose: {
    mop: xe
  },
  mpunct: {},
  minner: {
    mop: xe
  }
}, Ns = {}, Qr = {}, Jr = {};
function H(n) {
  for (var {
    type: e,
    names: t,
    props: r,
    handler: a,
    htmlBuilder: i,
    mathmlBuilder: l
  } = n, s = {
    type: e,
    numArgs: r.numArgs,
    argTypes: r.argTypes,
    allowedInArgument: !!r.allowedInArgument,
    allowedInText: !!r.allowedInText,
    allowedInMath: r.allowedInMath === void 0 ? !0 : r.allowedInMath,
    numOptionalArgs: r.numOptionalArgs || 0,
    infix: !!r.infix,
    primitive: !!r.primitive,
    handler: a
  }, u = 0; u < t.length; ++u)
    Ns[t[u]] = s;
  e && (i && (Qr[e] = i), l && (Jr[e] = l));
}
function x0(n) {
  var {
    type: e,
    htmlBuilder: t,
    mathmlBuilder: r
  } = n;
  H({
    type: e,
    names: [],
    props: {
      numArgs: 0
    },
    handler() {
      throw new Error("Should never be called.");
    },
    htmlBuilder: t,
    mathmlBuilder: r
  });
}
var en = function(e) {
  return e.type === "ordgroup" && e.body.length === 1 ? e.body[0] : e;
}, Fe = function(e) {
  return e.type === "ordgroup" ? e.body : [e];
}, n0 = F.makeSpan, P1 = ["leftmost", "mbin", "mopen", "mrel", "mop", "mpunct"], H1 = ["rightmost", "mrel", "mclose", "mpunct"], U1 = {
  display: Q.DISPLAY,
  text: Q.TEXT,
  script: Q.SCRIPT,
  scriptscript: Q.SCRIPTSCRIPT
}, G1 = {
  mord: "mord",
  mop: "mop",
  mbin: "mbin",
  mrel: "mrel",
  mopen: "mopen",
  mclose: "mclose",
  mpunct: "mpunct",
  minner: "minner"
}, Me = function(e, t, r, a) {
  a === void 0 && (a = [null, null]);
  for (var i = [], l = 0; l < e.length; l++) {
    var s = le(e[l], t);
    if (s instanceof ur) {
      var u = s.children;
      i.push(...u);
    } else
      i.push(s);
  }
  if (F.tryCombineChars(i), !r)
    return i;
  var h = t;
  if (e.length === 1) {
    var d = e[0];
    d.type === "sizing" ? h = t.havingSize(d.size) : d.type === "styling" && (h = t.havingStyle(U1[d.style]));
  }
  var g = n0([a[0] || "leftmost"], [], t), p = n0([a[1] || "rightmost"], [], t), v = r === "root";
  return Ri(i, (k, A) => {
    var C = A.classes[0], z = k.classes[0];
    C === "mbin" && Z.contains(H1, z) ? A.classes[0] = "mord" : z === "mbin" && Z.contains(P1, C) && (k.classes[0] = "mord");
  }, {
    node: g
  }, p, v), Ri(i, (k, A) => {
    var C = aa(A), z = aa(k), x = C && z ? k.hasClass("mtight") ? O1[C][z] : I1[C][z] : null;
    if (x)
      return F.makeGlue(x, h);
  }, {
    node: g
  }, p, v), i;
}, Ri = function n(e, t, r, a, i) {
  a && e.push(a);
  for (var l = 0; l < e.length; l++) {
    var s = e[l], u = qs(s);
    if (u) {
      n(u.children, t, r, null, i);
      continue;
    }
    var h = !s.hasClass("mspace");
    if (h) {
      var d = t(s, r.node);
      d && (r.insertAfter ? r.insertAfter(d) : (e.unshift(d), l++));
    }
    h ? r.node = s : i && s.hasClass("newline") && (r.node = n0(["leftmost"])), r.insertAfter = /* @__PURE__ */ ((g) => (p) => {
      e.splice(g + 1, 0, p), l++;
    })(l);
  }
  a && e.pop();
}, qs = function(e) {
  return e instanceof ur || e instanceof Aa || e instanceof cr && e.hasClass("enclosing") ? e : null;
}, V1 = function n(e, t) {
  var r = qs(e);
  if (r) {
    var a = r.children;
    if (a.length) {
      if (t === "right")
        return n(a[a.length - 1], "right");
      if (t === "left")
        return n(a[0], "left");
    }
  }
  return e;
}, aa = function(e, t) {
  return e ? (t && (e = V1(e, t)), G1[e.classes[0]] || null) : null;
}, or = function(e, t) {
  var r = ["nulldelimiter"].concat(e.baseSizingClasses());
  return n0(t.concat(r));
}, le = function(e, t, r) {
  if (!e)
    return n0();
  if (Qr[e.type]) {
    var a = Qr[e.type](e, t);
    if (r && t.size !== r.size) {
      a = n0(t.sizingClasses(r), [a], t);
      var i = t.sizeMultiplier / r.sizeMultiplier;
      a.height *= i, a.depth *= i;
    }
    return a;
  } else
    throw new L("Got group of unknown type: '" + e.type + "'");
};
function Dr(n, e) {
  var t = n0(["base"], n, e), r = n0(["strut"]);
  return r.style.height = P(t.height + t.depth), t.depth && (r.style.verticalAlign = P(-t.depth)), t.children.unshift(r), t;
}
function ia(n, e) {
  var t = null;
  n.length === 1 && n[0].type === "tag" && (t = n[0].tag, n = n[0].body);
  var r = Me(n, e, "root"), a;
  r.length === 2 && r[1].hasClass("tag") && (a = r.pop());
  for (var i = [], l = [], s = 0; s < r.length; s++)
    if (l.push(r[s]), r[s].hasClass("mbin") || r[s].hasClass("mrel") || r[s].hasClass("allowbreak")) {
      for (var u = !1; s < r.length - 1 && r[s + 1].hasClass("mspace") && !r[s + 1].hasClass("newline"); )
        s++, l.push(r[s]), r[s].hasClass("nobreak") && (u = !0);
      u || (i.push(Dr(l, e)), l = []);
    } else r[s].hasClass("newline") && (l.pop(), l.length > 0 && (i.push(Dr(l, e)), l = []), i.push(r[s]));
  l.length > 0 && i.push(Dr(l, e));
  var h;
  t ? (h = Dr(Me(t, e, !0)), h.classes = ["tag"], i.push(h)) : a && i.push(a);
  var d = n0(["katex-html"], i);
  if (d.setAttribute("aria-hidden", "true"), h) {
    var g = h.children[0];
    g.style.height = P(d.height + d.depth), d.depth && (g.style.verticalAlign = P(-d.depth));
  }
  return d;
}
function Ls(n) {
  return new ur(n);
}
class tt {
  constructor(e, t, r) {
    this.type = void 0, this.attributes = void 0, this.children = void 0, this.classes = void 0, this.type = e, this.attributes = {}, this.children = t || [], this.classes = r || [];
  }
  /**
   * Sets an attribute on a MathML node. MathML depends on attributes to convey a
   * semantic content, so this is used heavily.
   */
  setAttribute(e, t) {
    this.attributes[e] = t;
  }
  /**
   * Gets an attribute on a MathML node.
   */
  getAttribute(e) {
    return this.attributes[e];
  }
  /**
   * Converts the math node into a MathML-namespaced DOM element.
   */
  toNode() {
    var e = document.createElementNS("http://www.w3.org/1998/Math/MathML", this.type);
    for (var t in this.attributes)
      Object.prototype.hasOwnProperty.call(this.attributes, t) && e.setAttribute(t, this.attributes[t]);
    this.classes.length > 0 && (e.className = d0(this.classes));
    for (var r = 0; r < this.children.length; r++)
      if (this.children[r] instanceof Bt && this.children[r + 1] instanceof Bt) {
        for (var a = this.children[r].toText() + this.children[++r].toText(); this.children[r + 1] instanceof Bt; )
          a += this.children[++r].toText();
        e.appendChild(new Bt(a).toNode());
      } else
        e.appendChild(this.children[r].toNode());
    return e;
  }
  /**
   * Converts the math node into an HTML markup string.
   */
  toMarkup() {
    var e = "<" + this.type;
    for (var t in this.attributes)
      Object.prototype.hasOwnProperty.call(this.attributes, t) && (e += " " + t + '="', e += Z.escape(this.attributes[t]), e += '"');
    this.classes.length > 0 && (e += ' class ="' + Z.escape(d0(this.classes)) + '"'), e += ">";
    for (var r = 0; r < this.children.length; r++)
      e += this.children[r].toMarkup();
    return e += "</" + this.type + ">", e;
  }
  /**
   * Converts the math node into a string, similar to innerText, but escaped.
   */
  toText() {
    return this.children.map((e) => e.toText()).join("");
  }
}
class Bt {
  constructor(e) {
    this.text = void 0, this.text = e;
  }
  /**
   * Converts the text node into a DOM text node.
   */
  toNode() {
    return document.createTextNode(this.text);
  }
  /**
   * Converts the text node into escaped HTML markup
   * (representing the text itself).
   */
  toMarkup() {
    return Z.escape(this.toText());
  }
  /**
   * Converts the text node into a string
   * (representing the text itself).
   */
  toText() {
    return this.text;
  }
}
class W1 {
  /**
   * Create a Space node with width given in CSS ems.
   */
  constructor(e) {
    this.width = void 0, this.character = void 0, this.width = e, e >= 0.05555 && e <= 0.05556 ? this.character = " " : e >= 0.1666 && e <= 0.1667 ? this.character = " " : e >= 0.2222 && e <= 0.2223 ? this.character = " " : e >= 0.2777 && e <= 0.2778 ? this.character = "  " : e >= -0.05556 && e <= -0.05555 ? this.character = " ⁣" : e >= -0.1667 && e <= -0.1666 ? this.character = " ⁣" : e >= -0.2223 && e <= -0.2222 ? this.character = " ⁣" : e >= -0.2778 && e <= -0.2777 ? this.character = " ⁣" : this.character = null;
  }
  /**
   * Converts the math node into a MathML-namespaced DOM element.
   */
  toNode() {
    if (this.character)
      return document.createTextNode(this.character);
    var e = document.createElementNS("http://www.w3.org/1998/Math/MathML", "mspace");
    return e.setAttribute("width", P(this.width)), e;
  }
  /**
   * Converts the math node into an HTML markup string.
   */
  toMarkup() {
    return this.character ? "<mtext>" + this.character + "</mtext>" : '<mspace width="' + P(this.width) + '"/>';
  }
  /**
   * Converts the math node into a string, similar to innerText.
   */
  toText() {
    return this.character ? this.character : " ";
  }
}
var q = {
  MathNode: tt,
  TextNode: Bt,
  SpaceNode: W1,
  newDocumentFragment: Ls
}, ct = function(e, t, r) {
  return ge[t][e] && ge[t][e].replace && e.charCodeAt(0) !== 55349 && !($s.hasOwnProperty(e) && r && (r.fontFamily && r.fontFamily.slice(4, 6) === "tt" || r.font && r.font.slice(4, 6) === "tt")) && (e = ge[t][e].replace), new q.TextNode(e);
}, Fa = function(e) {
  return e.length === 1 ? e[0] : new q.MathNode("mrow", e);
}, Ca = function(e, t) {
  if (t.fontFamily === "texttt")
    return "monospace";
  if (t.fontFamily === "textsf")
    return t.fontShape === "textit" && t.fontWeight === "textbf" ? "sans-serif-bold-italic" : t.fontShape === "textit" ? "sans-serif-italic" : t.fontWeight === "textbf" ? "bold-sans-serif" : "sans-serif";
  if (t.fontShape === "textit" && t.fontWeight === "textbf")
    return "bold-italic";
  if (t.fontShape === "textit")
    return "italic";
  if (t.fontWeight === "textbf")
    return "bold";
  var r = t.font;
  if (!r || r === "mathnormal")
    return null;
  var a = e.mode;
  if (r === "mathit")
    return "italic";
  if (r === "boldsymbol")
    return e.type === "textord" ? "bold" : "bold-italic";
  if (r === "mathbf")
    return "bold";
  if (r === "mathbb")
    return "double-struck";
  if (r === "mathsfit")
    return "sans-serif-italic";
  if (r === "mathfrak")
    return "fraktur";
  if (r === "mathscr" || r === "mathcal")
    return "script";
  if (r === "mathsf")
    return "sans-serif";
  if (r === "mathtt")
    return "monospace";
  var i = e.text;
  if (Z.contains(["\\imath", "\\jmath"], i))
    return null;
  ge[a][i] && ge[a][i].replace && (i = ge[a][i].replace);
  var l = F.fontMap[r].fontName;
  return Sa(i, l, a) ? F.fontMap[r].variant : null;
};
function Cn(n) {
  if (!n)
    return !1;
  if (n.type === "mi" && n.children.length === 1) {
    var e = n.children[0];
    return e instanceof Bt && e.text === ".";
  } else if (n.type === "mo" && n.children.length === 1 && n.getAttribute("separator") === "true" && n.getAttribute("lspace") === "0em" && n.getAttribute("rspace") === "0em") {
    var t = n.children[0];
    return t instanceof Bt && t.text === ",";
  } else
    return !1;
}
var Ze = function(e, t, r) {
  if (e.length === 1) {
    var a = pe(e[0], t);
    return r && a instanceof tt && a.type === "mo" && (a.setAttribute("lspace", "0em"), a.setAttribute("rspace", "0em")), [a];
  }
  for (var i = [], l, s = 0; s < e.length; s++) {
    var u = pe(e[s], t);
    if (u instanceof tt && l instanceof tt) {
      if (u.type === "mtext" && l.type === "mtext" && u.getAttribute("mathvariant") === l.getAttribute("mathvariant")) {
        l.children.push(...u.children);
        continue;
      } else if (u.type === "mn" && l.type === "mn") {
        l.children.push(...u.children);
        continue;
      } else if (Cn(u) && l.type === "mn") {
        l.children.push(...u.children);
        continue;
      } else if (u.type === "mn" && Cn(l))
        u.children = [...l.children, ...u.children], i.pop();
      else if ((u.type === "msup" || u.type === "msub") && u.children.length >= 1 && (l.type === "mn" || Cn(l))) {
        var h = u.children[0];
        h instanceof tt && h.type === "mn" && (h.children = [...l.children, ...h.children], i.pop());
      } else if (l.type === "mi" && l.children.length === 1) {
        var d = l.children[0];
        if (d instanceof Bt && d.text === "̸" && (u.type === "mo" || u.type === "mi" || u.type === "mn")) {
          var g = u.children[0];
          g instanceof Bt && g.text.length > 0 && (g.text = g.text.slice(0, 1) + "̸" + g.text.slice(1), i.pop());
        }
      }
    }
    i.push(u), l = u;
  }
  return i;
}, f0 = function(e, t, r) {
  return Fa(Ze(e, t, r));
}, pe = function(e, t) {
  if (!e)
    return new q.MathNode("mrow");
  if (Jr[e.type]) {
    var r = Jr[e.type](e, t);
    return r;
  } else
    throw new L("Got group of unknown type: '" + e.type + "'");
};
function Ni(n, e, t, r, a) {
  var i = Ze(n, t), l;
  i.length === 1 && i[0] instanceof tt && Z.contains(["mrow", "mtable"], i[0].type) ? l = i[0] : l = new q.MathNode("mrow", i);
  var s = new q.MathNode("annotation", [new q.TextNode(e)]);
  s.setAttribute("encoding", "application/x-tex");
  var u = new q.MathNode("semantics", [l, s]), h = new q.MathNode("math", [u]);
  h.setAttribute("xmlns", "http://www.w3.org/1998/Math/MathML"), r && h.setAttribute("display", "block");
  var d = a ? "katex" : "katex-mathml";
  return F.makeSpan([d], [h]);
}
var Is = function(e) {
  return new Jt({
    style: e.displayMode ? Q.DISPLAY : Q.TEXT,
    maxSize: e.maxSize,
    minRuleThickness: e.minRuleThickness
  });
}, Os = function(e, t) {
  if (t.displayMode) {
    var r = ["katex-display"];
    t.leqno && r.push("leqno"), t.fleqn && r.push("fleqn"), e = F.makeSpan(r, [e]);
  }
  return e;
}, j1 = function(e, t, r) {
  var a = Is(r), i;
  if (r.output === "mathml")
    return Ni(e, t, a, r.displayMode, !0);
  if (r.output === "html") {
    var l = ia(e, a);
    i = F.makeSpan(["katex"], [l]);
  } else {
    var s = Ni(e, t, a, r.displayMode, !1), u = ia(e, a);
    i = F.makeSpan(["katex"], [s, u]);
  }
  return Os(i, r);
}, Y1 = function(e, t, r) {
  var a = Is(r), i = ia(e, a), l = F.makeSpan(["katex"], [i]);
  return Os(l, r);
}, X1 = {
  widehat: "^",
  widecheck: "ˇ",
  widetilde: "~",
  utilde: "~",
  overleftarrow: "←",
  underleftarrow: "←",
  xleftarrow: "←",
  overrightarrow: "→",
  underrightarrow: "→",
  xrightarrow: "→",
  underbrace: "⏟",
  overbrace: "⏞",
  overgroup: "⏠",
  undergroup: "⏡",
  overleftrightarrow: "↔",
  underleftrightarrow: "↔",
  xleftrightarrow: "↔",
  Overrightarrow: "⇒",
  xRightarrow: "⇒",
  overleftharpoon: "↼",
  xleftharpoonup: "↼",
  overrightharpoon: "⇀",
  xrightharpoonup: "⇀",
  xLeftarrow: "⇐",
  xLeftrightarrow: "⇔",
  xhookleftarrow: "↩",
  xhookrightarrow: "↪",
  xmapsto: "↦",
  xrightharpoondown: "⇁",
  xleftharpoondown: "↽",
  xrightleftharpoons: "⇌",
  xleftrightharpoons: "⇋",
  xtwoheadleftarrow: "↞",
  xtwoheadrightarrow: "↠",
  xlongequal: "=",
  xtofrom: "⇄",
  xrightleftarrows: "⇄",
  xrightequilibrium: "⇌",
  // Not a perfect match.
  xleftequilibrium: "⇋",
  // None better available.
  "\\cdrightarrow": "→",
  "\\cdleftarrow": "←",
  "\\cdlongequal": "="
}, Z1 = function(e) {
  var t = new q.MathNode("mo", [new q.TextNode(X1[e.replace(/^\\/, "")])]);
  return t.setAttribute("stretchy", "true"), t;
}, K1 = {
  //   path(s), minWidth, height, align
  overrightarrow: [["rightarrow"], 0.888, 522, "xMaxYMin"],
  overleftarrow: [["leftarrow"], 0.888, 522, "xMinYMin"],
  underrightarrow: [["rightarrow"], 0.888, 522, "xMaxYMin"],
  underleftarrow: [["leftarrow"], 0.888, 522, "xMinYMin"],
  xrightarrow: [["rightarrow"], 1.469, 522, "xMaxYMin"],
  "\\cdrightarrow": [["rightarrow"], 3, 522, "xMaxYMin"],
  // CD minwwidth2.5pc
  xleftarrow: [["leftarrow"], 1.469, 522, "xMinYMin"],
  "\\cdleftarrow": [["leftarrow"], 3, 522, "xMinYMin"],
  Overrightarrow: [["doublerightarrow"], 0.888, 560, "xMaxYMin"],
  xRightarrow: [["doublerightarrow"], 1.526, 560, "xMaxYMin"],
  xLeftarrow: [["doubleleftarrow"], 1.526, 560, "xMinYMin"],
  overleftharpoon: [["leftharpoon"], 0.888, 522, "xMinYMin"],
  xleftharpoonup: [["leftharpoon"], 0.888, 522, "xMinYMin"],
  xleftharpoondown: [["leftharpoondown"], 0.888, 522, "xMinYMin"],
  overrightharpoon: [["rightharpoon"], 0.888, 522, "xMaxYMin"],
  xrightharpoonup: [["rightharpoon"], 0.888, 522, "xMaxYMin"],
  xrightharpoondown: [["rightharpoondown"], 0.888, 522, "xMaxYMin"],
  xlongequal: [["longequal"], 0.888, 334, "xMinYMin"],
  "\\cdlongequal": [["longequal"], 3, 334, "xMinYMin"],
  xtwoheadleftarrow: [["twoheadleftarrow"], 0.888, 334, "xMinYMin"],
  xtwoheadrightarrow: [["twoheadrightarrow"], 0.888, 334, "xMaxYMin"],
  overleftrightarrow: [["leftarrow", "rightarrow"], 0.888, 522],
  overbrace: [["leftbrace", "midbrace", "rightbrace"], 1.6, 548],
  underbrace: [["leftbraceunder", "midbraceunder", "rightbraceunder"], 1.6, 548],
  underleftrightarrow: [["leftarrow", "rightarrow"], 0.888, 522],
  xleftrightarrow: [["leftarrow", "rightarrow"], 1.75, 522],
  xLeftrightarrow: [["doubleleftarrow", "doublerightarrow"], 1.75, 560],
  xrightleftharpoons: [["leftharpoondownplus", "rightharpoonplus"], 1.75, 716],
  xleftrightharpoons: [["leftharpoonplus", "rightharpoondownplus"], 1.75, 716],
  xhookleftarrow: [["leftarrow", "righthook"], 1.08, 522],
  xhookrightarrow: [["lefthook", "rightarrow"], 1.08, 522],
  overlinesegment: [["leftlinesegment", "rightlinesegment"], 0.888, 522],
  underlinesegment: [["leftlinesegment", "rightlinesegment"], 0.888, 522],
  overgroup: [["leftgroup", "rightgroup"], 0.888, 342],
  undergroup: [["leftgroupunder", "rightgroupunder"], 0.888, 342],
  xmapsto: [["leftmapsto", "rightarrow"], 1.5, 522],
  xtofrom: [["leftToFrom", "rightToFrom"], 1.75, 528],
  // The next three arrows are from the mhchem package.
  // In mhchem.sty, min-length is 2.0em. But these arrows might appear in the
  // document as \xrightarrow or \xrightleftharpoons. Those have
  // min-length = 1.75em, so we set min-length on these next three to match.
  xrightleftarrows: [["baraboveleftarrow", "rightarrowabovebar"], 1.75, 901],
  xrightequilibrium: [["baraboveshortleftharpoon", "rightharpoonaboveshortbar"], 1.75, 716],
  xleftequilibrium: [["shortbaraboveleftharpoon", "shortrightharpoonabovebar"], 1.75, 716]
}, Q1 = function(e) {
  return e.type === "ordgroup" ? e.body.length : 1;
}, J1 = function(e, t) {
  function r() {
    var s = 4e5, u = e.label.slice(1);
    if (Z.contains(["widehat", "widecheck", "widetilde", "utilde"], u)) {
      var h = e, d = Q1(h.base), g, p, v;
      if (d > 5)
        u === "widehat" || u === "widecheck" ? (g = 420, s = 2364, v = 0.42, p = u + "4") : (g = 312, s = 2340, v = 0.34, p = "tilde4");
      else {
        var k = [1, 1, 2, 2, 3, 3][d];
        u === "widehat" || u === "widecheck" ? (s = [0, 1062, 2364, 2364, 2364][k], g = [0, 239, 300, 360, 420][k], v = [0, 0.24, 0.3, 0.3, 0.36, 0.42][k], p = u + k) : (s = [0, 600, 1033, 2339, 2340][k], g = [0, 260, 286, 306, 312][k], v = [0, 0.26, 0.286, 0.3, 0.306, 0.34][k], p = "tilde" + k);
      }
      var A = new m0(p), C = new r0([A], {
        width: "100%",
        height: P(v),
        viewBox: "0 0 " + s + " " + g,
        preserveAspectRatio: "none"
      });
      return {
        span: F.makeSvgSpan([], [C], t),
        minWidth: 0,
        height: v
      };
    } else {
      var z = [], x = K1[u], [_, w, E] = x, T = E / 1e3, $ = _.length, M, B;
      if ($ === 1) {
        var G = x[3];
        M = ["hide-tail"], B = [G];
      } else if ($ === 2)
        M = ["halfarrow-left", "halfarrow-right"], B = ["xMinYMin", "xMaxYMin"];
      else if ($ === 3)
        M = ["brace-left", "brace-center", "brace-right"], B = ["xMinYMin", "xMidYMin", "xMaxYMin"];
      else
        throw new Error(`Correct katexImagesData or update code here to support
                    ` + $ + " children.");
      for (var U = 0; U < $; U++) {
        var j = new m0(_[U]), oe = new r0([j], {
          width: "400em",
          height: P(T),
          viewBox: "0 0 " + s + " " + E,
          preserveAspectRatio: B[U] + " slice"
        }), ee = F.makeSvgSpan([M[U]], [oe], t);
        if ($ === 1)
          return {
            span: ee,
            minWidth: w,
            height: T
          };
        ee.style.height = P(T), z.push(ee);
      }
      return {
        span: F.makeSpan(["stretchy"], z, t),
        minWidth: w,
        height: T
      };
    }
  }
  var {
    span: a,
    minWidth: i,
    height: l
  } = r();
  return a.height = l, a.style.height = P(l), i > 0 && (a.style.minWidth = P(i)), a;
}, ec = function(e, t, r, a, i) {
  var l, s = e.height + e.depth + r + a;
  if (/fbox|color|angl/.test(t)) {
    if (l = F.makeSpan(["stretchy", t], [], i), t === "fbox") {
      var u = i.color && i.getColor();
      u && (l.style.borderColor = u);
    }
  } else {
    var h = [];
    /^[bx]cancel$/.test(t) && h.push(new ra({
      x1: "0",
      y1: "0",
      x2: "100%",
      y2: "100%",
      "stroke-width": "0.046em"
    })), /^x?cancel$/.test(t) && h.push(new ra({
      x1: "0",
      y1: "100%",
      x2: "100%",
      y2: "0",
      "stroke-width": "0.046em"
    }));
    var d = new r0(h, {
      width: "100%",
      height: P(s)
    });
    l = F.makeSvgSpan([], [d], i);
  }
  return l.height = s, l.style.height = P(s), l;
}, a0 = {
  encloseSpan: ec,
  mathMLnode: Z1,
  svgSpan: J1
};
function re(n, e) {
  if (!n || n.type !== e)
    throw new Error("Expected node of type " + e + ", but got " + (n ? "node of type " + n.type : String(n)));
  return n;
}
function Ta(n) {
  var e = on(n);
  if (!e)
    throw new Error("Expected node of symbol group type, but got " + (n ? "node of type " + n.type : String(n)));
  return e;
}
function on(n) {
  return n && (n.type === "atom" || S1.hasOwnProperty(n.type)) ? n : null;
}
var $a = (n, e) => {
  var t, r, a;
  n && n.type === "supsub" ? (r = re(n.base, "accent"), t = r.base, n.base = t, a = k1(le(n, e)), n.base = r) : (r = re(n, "accent"), t = r.base);
  var i = le(t, e.havingCrampedStyle()), l = r.isShifty && Z.isCharacterBox(t), s = 0;
  if (l) {
    var u = Z.getBaseElem(t), h = le(u, e.havingCrampedStyle());
    s = Ci(h).skew;
  }
  var d = r.label === "\\c", g = d ? i.height + i.depth : Math.min(i.height, e.fontMetrics().xHeight), p;
  if (r.isStretchy)
    p = a0.svgSpan(r, e), p = F.makeVList({
      positionType: "firstBaseline",
      children: [{
        type: "elem",
        elem: i
      }, {
        type: "elem",
        elem: p,
        wrapperClasses: ["svg-align"],
        wrapperStyle: s > 0 ? {
          width: "calc(100% - " + P(2 * s) + ")",
          marginLeft: P(2 * s)
        } : void 0
      }]
    }, e);
  else {
    var v, k;
    r.label === "\\vec" ? (v = F.staticSvg("vec", e), k = F.svgData.vec[1]) : (v = F.makeOrd({
      mode: r.mode,
      text: r.label
    }, e, "textord"), v = Ci(v), v.italic = 0, k = v.width, d && (g += v.depth)), p = F.makeSpan(["accent-body"], [v]);
    var A = r.label === "\\textcircled";
    A && (p.classes.push("accent-full"), g = i.height);
    var C = s;
    A || (C -= k / 2), p.style.left = P(C), r.label === "\\textcircled" && (p.style.top = ".2em"), p = F.makeVList({
      positionType: "firstBaseline",
      children: [{
        type: "elem",
        elem: i
      }, {
        type: "kern",
        size: -g
      }, {
        type: "elem",
        elem: p
      }]
    }, e);
  }
  var z = F.makeSpan(["mord", "accent"], [p], e);
  return a ? (a.children[0] = z, a.height = Math.max(z.height, a.height), a.classes[0] = "mord", a) : z;
}, Ps = (n, e) => {
  var t = n.isStretchy ? a0.mathMLnode(n.label) : new q.MathNode("mo", [ct(n.label, n.mode)]), r = new q.MathNode("mover", [pe(n.base, e), t]);
  return r.setAttribute("accent", "true"), r;
}, tc = new RegExp(["\\acute", "\\grave", "\\ddot", "\\tilde", "\\bar", "\\breve", "\\check", "\\hat", "\\vec", "\\dot", "\\mathring"].map((n) => "\\" + n).join("|"));
H({
  type: "accent",
  names: ["\\acute", "\\grave", "\\ddot", "\\tilde", "\\bar", "\\breve", "\\check", "\\hat", "\\vec", "\\dot", "\\mathring", "\\widecheck", "\\widehat", "\\widetilde", "\\overrightarrow", "\\overleftarrow", "\\Overrightarrow", "\\overleftrightarrow", "\\overgroup", "\\overlinesegment", "\\overleftharpoon", "\\overrightharpoon"],
  props: {
    numArgs: 1
  },
  handler: (n, e) => {
    var t = en(e[0]), r = !tc.test(n.funcName), a = !r || n.funcName === "\\widehat" || n.funcName === "\\widetilde" || n.funcName === "\\widecheck";
    return {
      type: "accent",
      mode: n.parser.mode,
      label: n.funcName,
      isStretchy: r,
      isShifty: a,
      base: t
    };
  },
  htmlBuilder: $a,
  mathmlBuilder: Ps
});
H({
  type: "accent",
  names: ["\\'", "\\`", "\\^", "\\~", "\\=", "\\u", "\\.", '\\"', "\\c", "\\r", "\\H", "\\v", "\\textcircled"],
  props: {
    numArgs: 1,
    allowedInText: !0,
    allowedInMath: !0,
    // unless in strict mode
    argTypes: ["primitive"]
  },
  handler: (n, e) => {
    var t = e[0], r = n.parser.mode;
    return r === "math" && (n.parser.settings.reportNonstrict("mathVsTextAccents", "LaTeX's accent " + n.funcName + " works only in text mode"), r = "text"), {
      type: "accent",
      mode: r,
      label: n.funcName,
      isStretchy: !1,
      isShifty: !0,
      base: t
    };
  },
  htmlBuilder: $a,
  mathmlBuilder: Ps
});
H({
  type: "accentUnder",
  names: ["\\underleftarrow", "\\underrightarrow", "\\underleftrightarrow", "\\undergroup", "\\underlinesegment", "\\utilde"],
  props: {
    numArgs: 1
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0];
    return {
      type: "accentUnder",
      mode: t.mode,
      label: r,
      base: a
    };
  },
  htmlBuilder: (n, e) => {
    var t = le(n.base, e), r = a0.svgSpan(n, e), a = n.label === "\\utilde" ? 0.12 : 0, i = F.makeVList({
      positionType: "top",
      positionData: t.height,
      children: [{
        type: "elem",
        elem: r,
        wrapperClasses: ["svg-align"]
      }, {
        type: "kern",
        size: a
      }, {
        type: "elem",
        elem: t
      }]
    }, e);
    return F.makeSpan(["mord", "accentunder"], [i], e);
  },
  mathmlBuilder: (n, e) => {
    var t = a0.mathMLnode(n.label), r = new q.MathNode("munder", [pe(n.base, e), t]);
    return r.setAttribute("accentunder", "true"), r;
  }
});
var Sr = (n) => {
  var e = new q.MathNode("mpadded", n ? [n] : []);
  return e.setAttribute("width", "+0.6em"), e.setAttribute("lspace", "0.3em"), e;
};
H({
  type: "xArrow",
  names: [
    "\\xleftarrow",
    "\\xrightarrow",
    "\\xLeftarrow",
    "\\xRightarrow",
    "\\xleftrightarrow",
    "\\xLeftrightarrow",
    "\\xhookleftarrow",
    "\\xhookrightarrow",
    "\\xmapsto",
    "\\xrightharpoondown",
    "\\xrightharpoonup",
    "\\xleftharpoondown",
    "\\xleftharpoonup",
    "\\xrightleftharpoons",
    "\\xleftrightharpoons",
    "\\xlongequal",
    "\\xtwoheadrightarrow",
    "\\xtwoheadleftarrow",
    "\\xtofrom",
    // The next 3 functions are here to support the mhchem extension.
    // Direct use of these functions is discouraged and may break someday.
    "\\xrightleftarrows",
    "\\xrightequilibrium",
    "\\xleftequilibrium",
    // The next 3 functions are here only to support the {CD} environment.
    "\\\\cdrightarrow",
    "\\\\cdleftarrow",
    "\\\\cdlongequal"
  ],
  props: {
    numArgs: 1,
    numOptionalArgs: 1
  },
  handler(n, e, t) {
    var {
      parser: r,
      funcName: a
    } = n;
    return {
      type: "xArrow",
      mode: r.mode,
      label: a,
      body: e[0],
      below: t[0]
    };
  },
  // Flow is unable to correctly infer the type of `group`, even though it's
  // unambiguously determined from the passed-in `type` above.
  htmlBuilder(n, e) {
    var t = e.style, r = e.havingStyle(t.sup()), a = F.wrapFragment(le(n.body, r, e), e), i = n.label.slice(0, 2) === "\\x" ? "x" : "cd";
    a.classes.push(i + "-arrow-pad");
    var l;
    n.below && (r = e.havingStyle(t.sub()), l = F.wrapFragment(le(n.below, r, e), e), l.classes.push(i + "-arrow-pad"));
    var s = a0.svgSpan(n, e), u = -e.fontMetrics().axisHeight + 0.5 * s.height, h = -e.fontMetrics().axisHeight - 0.5 * s.height - 0.111;
    (a.depth > 0.25 || n.label === "\\xleftequilibrium") && (h -= a.depth);
    var d;
    if (l) {
      var g = -e.fontMetrics().axisHeight + l.height + 0.5 * s.height + 0.111;
      d = F.makeVList({
        positionType: "individualShift",
        children: [{
          type: "elem",
          elem: a,
          shift: h
        }, {
          type: "elem",
          elem: s,
          shift: u
        }, {
          type: "elem",
          elem: l,
          shift: g
        }]
      }, e);
    } else
      d = F.makeVList({
        positionType: "individualShift",
        children: [{
          type: "elem",
          elem: a,
          shift: h
        }, {
          type: "elem",
          elem: s,
          shift: u
        }]
      }, e);
    return d.children[0].children[0].children[1].classes.push("svg-align"), F.makeSpan(["mrel", "x-arrow"], [d], e);
  },
  mathmlBuilder(n, e) {
    var t = a0.mathMLnode(n.label);
    t.setAttribute("minsize", n.label.charAt(0) === "x" ? "1.75em" : "3.0em");
    var r;
    if (n.body) {
      var a = Sr(pe(n.body, e));
      if (n.below) {
        var i = Sr(pe(n.below, e));
        r = new q.MathNode("munderover", [t, i, a]);
      } else
        r = new q.MathNode("mover", [t, a]);
    } else if (n.below) {
      var l = Sr(pe(n.below, e));
      r = new q.MathNode("munder", [t, l]);
    } else
      r = Sr(), r = new q.MathNode("mover", [t, r]);
    return r;
  }
});
var rc = F.makeSpan;
function Hs(n, e) {
  var t = Me(n.body, e, !0);
  return rc([n.mclass], t, e);
}
function Us(n, e) {
  var t, r = Ze(n.body, e);
  return n.mclass === "minner" ? t = new q.MathNode("mpadded", r) : n.mclass === "mord" ? n.isCharacterBox ? (t = r[0], t.type = "mi") : t = new q.MathNode("mi", r) : (n.isCharacterBox ? (t = r[0], t.type = "mo") : t = new q.MathNode("mo", r), n.mclass === "mbin" ? (t.attributes.lspace = "0.22em", t.attributes.rspace = "0.22em") : n.mclass === "mpunct" ? (t.attributes.lspace = "0em", t.attributes.rspace = "0.17em") : n.mclass === "mopen" || n.mclass === "mclose" ? (t.attributes.lspace = "0em", t.attributes.rspace = "0em") : n.mclass === "minner" && (t.attributes.lspace = "0.0556em", t.attributes.width = "+0.1111em")), t;
}
H({
  type: "mclass",
  names: ["\\mathord", "\\mathbin", "\\mathrel", "\\mathopen", "\\mathclose", "\\mathpunct", "\\mathinner"],
  props: {
    numArgs: 1,
    primitive: !0
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0];
    return {
      type: "mclass",
      mode: t.mode,
      mclass: "m" + r.slice(5),
      // TODO(kevinb): don't prefix with 'm'
      body: Fe(a),
      isCharacterBox: Z.isCharacterBox(a)
    };
  },
  htmlBuilder: Hs,
  mathmlBuilder: Us
});
var un = (n) => {
  var e = n.type === "ordgroup" && n.body.length ? n.body[0] : n;
  return e.type === "atom" && (e.family === "bin" || e.family === "rel") ? "m" + e.family : "mord";
};
H({
  type: "mclass",
  names: ["\\@binrel"],
  props: {
    numArgs: 2
  },
  handler(n, e) {
    var {
      parser: t
    } = n;
    return {
      type: "mclass",
      mode: t.mode,
      mclass: un(e[0]),
      body: Fe(e[1]),
      isCharacterBox: Z.isCharacterBox(e[1])
    };
  }
});
H({
  type: "mclass",
  names: ["\\stackrel", "\\overset", "\\underset"],
  props: {
    numArgs: 2
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r
    } = n, a = e[1], i = e[0], l;
    r !== "\\stackrel" ? l = un(a) : l = "mrel";
    var s = {
      type: "op",
      mode: a.mode,
      limits: !0,
      alwaysHandleSupSub: !0,
      parentIsSupSub: !1,
      symbol: !1,
      suppressBaseShift: r !== "\\stackrel",
      body: Fe(a)
    }, u = {
      type: "supsub",
      mode: i.mode,
      base: s,
      sup: r === "\\underset" ? null : i,
      sub: r === "\\underset" ? i : null
    };
    return {
      type: "mclass",
      mode: t.mode,
      mclass: l,
      body: [u],
      isCharacterBox: Z.isCharacterBox(u)
    };
  },
  htmlBuilder: Hs,
  mathmlBuilder: Us
});
H({
  type: "pmb",
  names: ["\\pmb"],
  props: {
    numArgs: 1,
    allowedInText: !0
  },
  handler(n, e) {
    var {
      parser: t
    } = n;
    return {
      type: "pmb",
      mode: t.mode,
      mclass: un(e[0]),
      body: Fe(e[0])
    };
  },
  htmlBuilder(n, e) {
    var t = Me(n.body, e, !0), r = F.makeSpan([n.mclass], t, e);
    return r.style.textShadow = "0.02em 0.01em 0.04px", r;
  },
  mathmlBuilder(n, e) {
    var t = Ze(n.body, e), r = new q.MathNode("mstyle", t);
    return r.setAttribute("style", "text-shadow: 0.02em 0.01em 0.04px"), r;
  }
});
var nc = {
  ">": "\\\\cdrightarrow",
  "<": "\\\\cdleftarrow",
  "=": "\\\\cdlongequal",
  A: "\\uparrow",
  V: "\\downarrow",
  "|": "\\Vert",
  ".": "no arrow"
}, qi = () => ({
  type: "styling",
  body: [],
  mode: "math",
  style: "display"
}), Li = (n) => n.type === "textord" && n.text === "@", ac = (n, e) => (n.type === "mathord" || n.type === "atom") && n.text === e;
function ic(n, e, t) {
  var r = nc[n];
  switch (r) {
    case "\\\\cdrightarrow":
    case "\\\\cdleftarrow":
      return t.callFunction(r, [e[0]], [e[1]]);
    case "\\uparrow":
    case "\\downarrow": {
      var a = t.callFunction("\\\\cdleft", [e[0]], []), i = {
        type: "atom",
        text: r,
        mode: "math",
        family: "rel"
      }, l = t.callFunction("\\Big", [i], []), s = t.callFunction("\\\\cdright", [e[1]], []), u = {
        type: "ordgroup",
        mode: "math",
        body: [a, l, s]
      };
      return t.callFunction("\\\\cdparent", [u], []);
    }
    case "\\\\cdlongequal":
      return t.callFunction("\\\\cdlongequal", [], []);
    case "\\Vert": {
      var h = {
        type: "textord",
        text: "\\Vert",
        mode: "math"
      };
      return t.callFunction("\\Big", [h], []);
    }
    default:
      return {
        type: "textord",
        text: " ",
        mode: "math"
      };
  }
}
function lc(n) {
  var e = [];
  for (n.gullet.beginGroup(), n.gullet.macros.set("\\cr", "\\\\\\relax"), n.gullet.beginGroup(); ; ) {
    e.push(n.parseExpression(!1, "\\\\")), n.gullet.endGroup(), n.gullet.beginGroup();
    var t = n.fetch().text;
    if (t === "&" || t === "\\\\")
      n.consume();
    else if (t === "\\end") {
      e[e.length - 1].length === 0 && e.pop();
      break;
    } else
      throw new L("Expected \\\\ or \\cr or \\end", n.nextToken);
  }
  for (var r = [], a = [r], i = 0; i < e.length; i++) {
    for (var l = e[i], s = qi(), u = 0; u < l.length; u++)
      if (!Li(l[u]))
        s.body.push(l[u]);
      else {
        r.push(s), u += 1;
        var h = Ta(l[u]).text, d = new Array(2);
        if (d[0] = {
          type: "ordgroup",
          mode: "math",
          body: []
        }, d[1] = {
          type: "ordgroup",
          mode: "math",
          body: []
        }, !("=|.".indexOf(h) > -1)) if ("<>AV".indexOf(h) > -1)
          for (var g = 0; g < 2; g++) {
            for (var p = !0, v = u + 1; v < l.length; v++) {
              if (ac(l[v], h)) {
                p = !1, u = v;
                break;
              }
              if (Li(l[v]))
                throw new L("Missing a " + h + " character to complete a CD arrow.", l[v]);
              d[g].body.push(l[v]);
            }
            if (p)
              throw new L("Missing a " + h + " character to complete a CD arrow.", l[u]);
          }
        else
          throw new L('Expected one of "<>AV=|." after @', l[u]);
        var k = ic(h, d, n), A = {
          type: "styling",
          body: [k],
          mode: "math",
          style: "display"
          // CD is always displaystyle.
        };
        r.push(A), s = qi();
      }
    i % 2 === 0 ? r.push(s) : r.shift(), r = [], a.push(r);
  }
  n.gullet.endGroup(), n.gullet.endGroup();
  var C = new Array(a[0].length).fill({
    type: "align",
    align: "c",
    pregap: 0.25,
    // CD package sets \enskip between columns.
    postgap: 0.25
    // So pre and post each get half an \enskip, i.e. 0.25em.
  });
  return {
    type: "array",
    mode: "math",
    body: a,
    arraystretch: 1,
    addJot: !0,
    rowGaps: [null],
    cols: C,
    colSeparationType: "CD",
    hLinesBeforeRow: new Array(a.length + 1).fill([])
  };
}
H({
  type: "cdlabel",
  names: ["\\\\cdleft", "\\\\cdright"],
  props: {
    numArgs: 1
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r
    } = n;
    return {
      type: "cdlabel",
      mode: t.mode,
      side: r.slice(4),
      label: e[0]
    };
  },
  htmlBuilder(n, e) {
    var t = e.havingStyle(e.style.sup()), r = F.wrapFragment(le(n.label, t, e), e);
    return r.classes.push("cd-label-" + n.side), r.style.bottom = P(0.8 - r.depth), r.height = 0, r.depth = 0, r;
  },
  mathmlBuilder(n, e) {
    var t = new q.MathNode("mrow", [pe(n.label, e)]);
    return t = new q.MathNode("mpadded", [t]), t.setAttribute("width", "0"), n.side === "left" && t.setAttribute("lspace", "-1width"), t.setAttribute("voffset", "0.7em"), t = new q.MathNode("mstyle", [t]), t.setAttribute("displaystyle", "false"), t.setAttribute("scriptlevel", "1"), t;
  }
});
H({
  type: "cdlabelparent",
  names: ["\\\\cdparent"],
  props: {
    numArgs: 1
  },
  handler(n, e) {
    var {
      parser: t
    } = n;
    return {
      type: "cdlabelparent",
      mode: t.mode,
      fragment: e[0]
    };
  },
  htmlBuilder(n, e) {
    var t = F.wrapFragment(le(n.fragment, e), e);
    return t.classes.push("cd-vert-arrow"), t;
  },
  mathmlBuilder(n, e) {
    return new q.MathNode("mrow", [pe(n.fragment, e)]);
  }
});
H({
  type: "textord",
  names: ["\\@char"],
  props: {
    numArgs: 1,
    allowedInText: !0
  },
  handler(n, e) {
    for (var {
      parser: t
    } = n, r = re(e[0], "ordgroup"), a = r.body, i = "", l = 0; l < a.length; l++) {
      var s = re(a[l], "textord");
      i += s.text;
    }
    var u = parseInt(i), h;
    if (isNaN(u))
      throw new L("\\@char has non-numeric argument " + i);
    if (u < 0 || u >= 1114111)
      throw new L("\\@char with invalid code point " + i);
    return u <= 65535 ? h = String.fromCharCode(u) : (u -= 65536, h = String.fromCharCode((u >> 10) + 55296, (u & 1023) + 56320)), {
      type: "textord",
      mode: t.mode,
      text: h
    };
  }
});
var Gs = (n, e) => {
  var t = Me(n.body, e.withColor(n.color), !1);
  return F.makeFragment(t);
}, Vs = (n, e) => {
  var t = Ze(n.body, e.withColor(n.color)), r = new q.MathNode("mstyle", t);
  return r.setAttribute("mathcolor", n.color), r;
};
H({
  type: "color",
  names: ["\\textcolor"],
  props: {
    numArgs: 2,
    allowedInText: !0,
    argTypes: ["color", "original"]
  },
  handler(n, e) {
    var {
      parser: t
    } = n, r = re(e[0], "color-token").color, a = e[1];
    return {
      type: "color",
      mode: t.mode,
      color: r,
      body: Fe(a)
    };
  },
  htmlBuilder: Gs,
  mathmlBuilder: Vs
});
H({
  type: "color",
  names: ["\\color"],
  props: {
    numArgs: 1,
    allowedInText: !0,
    argTypes: ["color"]
  },
  handler(n, e) {
    var {
      parser: t,
      breakOnTokenText: r
    } = n, a = re(e[0], "color-token").color;
    t.gullet.macros.set("\\current@color", a);
    var i = t.parseExpression(!0, r);
    return {
      type: "color",
      mode: t.mode,
      color: a,
      body: i
    };
  },
  htmlBuilder: Gs,
  mathmlBuilder: Vs
});
H({
  type: "cr",
  names: ["\\\\"],
  props: {
    numArgs: 0,
    numOptionalArgs: 0,
    allowedInText: !0
  },
  handler(n, e, t) {
    var {
      parser: r
    } = n, a = r.gullet.future().text === "[" ? r.parseSizeGroup(!0) : null, i = !r.settings.displayMode || !r.settings.useStrictBehavior("newLineInDisplayMode", "In LaTeX, \\\\ or \\newline does nothing in display mode");
    return {
      type: "cr",
      mode: r.mode,
      newLine: i,
      size: a && re(a, "size").value
    };
  },
  // The following builders are called only at the top level,
  // not within tabular/array environments.
  htmlBuilder(n, e) {
    var t = F.makeSpan(["mspace"], [], e);
    return n.newLine && (t.classes.push("newline"), n.size && (t.style.marginTop = P(ke(n.size, e)))), t;
  },
  mathmlBuilder(n, e) {
    var t = new q.MathNode("mspace");
    return n.newLine && (t.setAttribute("linebreak", "newline"), n.size && t.setAttribute("height", P(ke(n.size, e)))), t;
  }
});
var la = {
  "\\global": "\\global",
  "\\long": "\\\\globallong",
  "\\\\globallong": "\\\\globallong",
  "\\def": "\\gdef",
  "\\gdef": "\\gdef",
  "\\edef": "\\xdef",
  "\\xdef": "\\xdef",
  "\\let": "\\\\globallet",
  "\\futurelet": "\\\\globalfuture"
}, Ws = (n) => {
  var e = n.text;
  if (/^(?:[\\{}$&#^_]|EOF)$/.test(e))
    throw new L("Expected a control sequence", n);
  return e;
}, sc = (n) => {
  var e = n.gullet.popToken();
  return e.text === "=" && (e = n.gullet.popToken(), e.text === " " && (e = n.gullet.popToken())), e;
}, js = (n, e, t, r) => {
  var a = n.gullet.macros.get(t.text);
  a == null && (t.noexpand = !0, a = {
    tokens: [t],
    numArgs: 0,
    // reproduce the same behavior in expansion
    unexpandable: !n.gullet.isExpandable(t.text)
  }), n.gullet.macros.set(e, a, r);
};
H({
  type: "internal",
  names: [
    "\\global",
    "\\long",
    "\\\\globallong"
    // can’t be entered directly
  ],
  props: {
    numArgs: 0,
    allowedInText: !0
  },
  handler(n) {
    var {
      parser: e,
      funcName: t
    } = n;
    e.consumeSpaces();
    var r = e.fetch();
    if (la[r.text])
      return (t === "\\global" || t === "\\\\globallong") && (r.text = la[r.text]), re(e.parseFunction(), "internal");
    throw new L("Invalid token after macro prefix", r);
  }
});
H({
  type: "internal",
  names: ["\\def", "\\gdef", "\\edef", "\\xdef"],
  props: {
    numArgs: 0,
    allowedInText: !0,
    primitive: !0
  },
  handler(n) {
    var {
      parser: e,
      funcName: t
    } = n, r = e.gullet.popToken(), a = r.text;
    if (/^(?:[\\{}$&#^_]|EOF)$/.test(a))
      throw new L("Expected a control sequence", r);
    for (var i = 0, l, s = [[]]; e.gullet.future().text !== "{"; )
      if (r = e.gullet.popToken(), r.text === "#") {
        if (e.gullet.future().text === "{") {
          l = e.gullet.future(), s[i].push("{");
          break;
        }
        if (r = e.gullet.popToken(), !/^[1-9]$/.test(r.text))
          throw new L('Invalid argument number "' + r.text + '"');
        if (parseInt(r.text) !== i + 1)
          throw new L('Argument number "' + r.text + '" out of order');
        i++, s.push([]);
      } else {
        if (r.text === "EOF")
          throw new L("Expected a macro definition");
        s[i].push(r.text);
      }
    var {
      tokens: u
    } = e.gullet.consumeArg();
    return l && u.unshift(l), (t === "\\edef" || t === "\\xdef") && (u = e.gullet.expandTokens(u), u.reverse()), e.gullet.macros.set(a, {
      tokens: u,
      numArgs: i,
      delimiters: s
    }, t === la[t]), {
      type: "internal",
      mode: e.mode
    };
  }
});
H({
  type: "internal",
  names: [
    "\\let",
    "\\\\globallet"
    // can’t be entered directly
  ],
  props: {
    numArgs: 0,
    allowedInText: !0,
    primitive: !0
  },
  handler(n) {
    var {
      parser: e,
      funcName: t
    } = n, r = Ws(e.gullet.popToken());
    e.gullet.consumeSpaces();
    var a = sc(e);
    return js(e, r, a, t === "\\\\globallet"), {
      type: "internal",
      mode: e.mode
    };
  }
});
H({
  type: "internal",
  names: [
    "\\futurelet",
    "\\\\globalfuture"
    // can’t be entered directly
  ],
  props: {
    numArgs: 0,
    allowedInText: !0,
    primitive: !0
  },
  handler(n) {
    var {
      parser: e,
      funcName: t
    } = n, r = Ws(e.gullet.popToken()), a = e.gullet.popToken(), i = e.gullet.popToken();
    return js(e, r, i, t === "\\\\globalfuture"), e.gullet.pushToken(i), e.gullet.pushToken(a), {
      type: "internal",
      mode: e.mode
    };
  }
});
var K0 = function(e, t, r) {
  var a = ge.math[e] && ge.math[e].replace, i = Sa(a || e, t, r);
  if (!i)
    throw new Error("Unsupported symbol " + e + " and font size " + t + ".");
  return i;
}, Ma = function(e, t, r, a) {
  var i = r.havingBaseStyle(t), l = F.makeSpan(a.concat(i.sizingClasses(r)), [e], r), s = i.sizeMultiplier / r.sizeMultiplier;
  return l.height *= s, l.depth *= s, l.maxFontSize = i.sizeMultiplier, l;
}, Ys = function(e, t, r) {
  var a = t.havingBaseStyle(r), i = (1 - t.sizeMultiplier / a.sizeMultiplier) * t.fontMetrics().axisHeight;
  e.classes.push("delimcenter"), e.style.top = P(i), e.height -= i, e.depth += i;
}, oc = function(e, t, r, a, i, l) {
  var s = F.makeSymbol(e, "Main-Regular", i, a), u = Ma(s, t, a, l);
  return r && Ys(u, a, t), u;
}, uc = function(e, t, r, a) {
  return F.makeSymbol(e, "Size" + t + "-Regular", r, a);
}, Xs = function(e, t, r, a, i, l) {
  var s = uc(e, t, i, a), u = Ma(F.makeSpan(["delimsizing", "size" + t], [s], a), Q.TEXT, a, l);
  return r && Ys(u, a, Q.TEXT), u;
}, Tn = function(e, t, r) {
  var a;
  t === "Size1-Regular" ? a = "delim-size1" : a = "delim-size4";
  var i = F.makeSpan(["delimsizinginner", a], [F.makeSpan([], [F.makeSymbol(e, t, r)])]);
  return {
    type: "elem",
    elem: i
  };
}, $n = function(e, t, r) {
  var a = zt["Size4-Regular"][e.charCodeAt(0)] ? zt["Size4-Regular"][e.charCodeAt(0)][4] : zt["Size1-Regular"][e.charCodeAt(0)][4], i = new m0("inner", p1(e, Math.round(1e3 * t))), l = new r0([i], {
    width: P(a),
    height: P(t),
    // Override CSS rule `.katex svg { width: 100% }`
    style: "width:" + P(a),
    viewBox: "0 0 " + 1e3 * a + " " + Math.round(1e3 * t),
    preserveAspectRatio: "xMinYMin"
  }), s = F.makeSvgSpan([], [l], r);
  return s.height = t, s.style.height = P(t), s.style.width = P(a), {
    type: "elem",
    elem: s
  };
}, sa = 8e-3, Ar = {
  type: "kern",
  size: -1 * sa
}, cc = ["|", "\\lvert", "\\rvert", "\\vert"], hc = ["\\|", "\\lVert", "\\rVert", "\\Vert"], Zs = function(e, t, r, a, i, l) {
  var s, u, h, d, g = "", p = 0;
  s = h = d = e, u = null;
  var v = "Size1-Regular";
  e === "\\uparrow" ? h = d = "⏐" : e === "\\Uparrow" ? h = d = "‖" : e === "\\downarrow" ? s = h = "⏐" : e === "\\Downarrow" ? s = h = "‖" : e === "\\updownarrow" ? (s = "\\uparrow", h = "⏐", d = "\\downarrow") : e === "\\Updownarrow" ? (s = "\\Uparrow", h = "‖", d = "\\Downarrow") : Z.contains(cc, e) ? (h = "∣", g = "vert", p = 333) : Z.contains(hc, e) ? (h = "∥", g = "doublevert", p = 556) : e === "[" || e === "\\lbrack" ? (s = "⎡", h = "⎢", d = "⎣", v = "Size4-Regular", g = "lbrack", p = 667) : e === "]" || e === "\\rbrack" ? (s = "⎤", h = "⎥", d = "⎦", v = "Size4-Regular", g = "rbrack", p = 667) : e === "\\lfloor" || e === "⌊" ? (h = s = "⎢", d = "⎣", v = "Size4-Regular", g = "lfloor", p = 667) : e === "\\lceil" || e === "⌈" ? (s = "⎡", h = d = "⎢", v = "Size4-Regular", g = "lceil", p = 667) : e === "\\rfloor" || e === "⌋" ? (h = s = "⎥", d = "⎦", v = "Size4-Regular", g = "rfloor", p = 667) : e === "\\rceil" || e === "⌉" ? (s = "⎤", h = d = "⎥", v = "Size4-Regular", g = "rceil", p = 667) : e === "(" || e === "\\lparen" ? (s = "⎛", h = "⎜", d = "⎝", v = "Size4-Regular", g = "lparen", p = 875) : e === ")" || e === "\\rparen" ? (s = "⎞", h = "⎟", d = "⎠", v = "Size4-Regular", g = "rparen", p = 875) : e === "\\{" || e === "\\lbrace" ? (s = "⎧", u = "⎨", d = "⎩", h = "⎪", v = "Size4-Regular") : e === "\\}" || e === "\\rbrace" ? (s = "⎫", u = "⎬", d = "⎭", h = "⎪", v = "Size4-Regular") : e === "\\lgroup" || e === "⟮" ? (s = "⎧", d = "⎩", h = "⎪", v = "Size4-Regular") : e === "\\rgroup" || e === "⟯" ? (s = "⎫", d = "⎭", h = "⎪", v = "Size4-Regular") : e === "\\lmoustache" || e === "⎰" ? (s = "⎧", d = "⎭", h = "⎪", v = "Size4-Regular") : (e === "\\rmoustache" || e === "⎱") && (s = "⎫", d = "⎩", h = "⎪", v = "Size4-Regular");
  var k = K0(s, v, i), A = k.height + k.depth, C = K0(h, v, i), z = C.height + C.depth, x = K0(d, v, i), _ = x.height + x.depth, w = 0, E = 1;
  if (u !== null) {
    var T = K0(u, v, i);
    w = T.height + T.depth, E = 2;
  }
  var $ = A + _ + w, M = Math.max(0, Math.ceil((t - $) / (E * z))), B = $ + M * E * z, G = a.fontMetrics().axisHeight;
  r && (G *= a.sizeMultiplier);
  var U = B / 2 - G, j = [];
  if (g.length > 0) {
    var oe = B - A - _, ee = Math.round(B * 1e3), ue = g1(g, Math.round(oe * 1e3)), fe = new m0(g, ue), Ee = (p / 1e3).toFixed(3) + "em", ne = (ee / 1e3).toFixed(3) + "em", ve = new r0([fe], {
      width: Ee,
      height: ne,
      viewBox: "0 0 " + p + " " + ee
    }), we = F.makeSvgSpan([], [ve], a);
    we.height = ee / 1e3, we.style.width = Ee, we.style.height = ne, j.push({
      type: "elem",
      elem: we
    });
  } else {
    if (j.push(Tn(d, v, i)), j.push(Ar), u === null) {
      var N = B - A - _ + 2 * sa;
      j.push($n(h, N, a));
    } else {
      var se = (B - A - _ - w) / 2 + 2 * sa;
      j.push($n(h, se, a)), j.push(Ar), j.push(Tn(u, v, i)), j.push(Ar), j.push($n(h, se, a));
    }
    j.push(Ar), j.push(Tn(s, v, i));
  }
  var ce = a.havingBaseStyle(Q.TEXT), Ce = F.makeVList({
    positionType: "bottom",
    positionData: U,
    children: j
  }, ce);
  return Ma(F.makeSpan(["delimsizing", "mult"], [Ce], ce), Q.TEXT, a, l);
}, Mn = 80, zn = 0.08, Bn = function(e, t, r, a, i) {
  var l = f1(e, a, r), s = new m0(e, l), u = new r0([s], {
    // Note: 1000:1 ratio of viewBox to document em width.
    width: "400em",
    height: P(t),
    viewBox: "0 0 400000 " + r,
    preserveAspectRatio: "xMinYMin slice"
  });
  return F.makeSvgSpan(["hide-tail"], [u], i);
}, dc = function(e, t) {
  var r = t.havingBaseSizing(), a = eo("\\surd", e * r.sizeMultiplier, Js, r), i = r.sizeMultiplier, l = Math.max(0, t.minRuleThickness - t.fontMetrics().sqrtRuleThickness), s, u = 0, h = 0, d = 0, g;
  return a.type === "small" ? (d = 1e3 + 1e3 * l + Mn, e < 1 ? i = 1 : e < 1.4 && (i = 0.7), u = (1 + l + zn) / i, h = (1 + l) / i, s = Bn("sqrtMain", u, d, l, t), s.style.minWidth = "0.853em", g = 0.833 / i) : a.type === "large" ? (d = (1e3 + Mn) * tr[a.size], h = (tr[a.size] + l) / i, u = (tr[a.size] + l + zn) / i, s = Bn("sqrtSize" + a.size, u, d, l, t), s.style.minWidth = "1.02em", g = 1 / i) : (u = e + l + zn, h = e + l, d = Math.floor(1e3 * e + l) + Mn, s = Bn("sqrtTall", u, d, l, t), s.style.minWidth = "0.742em", g = 1.056), s.height = h, s.style.height = P(u), {
    span: s,
    advanceWidth: g,
    // Calculate the actual line width.
    // This actually should depend on the chosen font -- e.g. \boldmath
    // should use the thicker surd symbols from e.g. KaTeX_Main-Bold, and
    // have thicker rules.
    ruleWidth: (t.fontMetrics().sqrtRuleThickness + l) * i
  };
}, Ks = ["(", "\\lparen", ")", "\\rparen", "[", "\\lbrack", "]", "\\rbrack", "\\{", "\\lbrace", "\\}", "\\rbrace", "\\lfloor", "\\rfloor", "⌊", "⌋", "\\lceil", "\\rceil", "⌈", "⌉", "\\surd"], mc = ["\\uparrow", "\\downarrow", "\\updownarrow", "\\Uparrow", "\\Downarrow", "\\Updownarrow", "|", "\\|", "\\vert", "\\Vert", "\\lvert", "\\rvert", "\\lVert", "\\rVert", "\\lgroup", "\\rgroup", "⟮", "⟯", "\\lmoustache", "\\rmoustache", "⎰", "⎱"], Qs = ["<", ">", "\\langle", "\\rangle", "/", "\\backslash", "\\lt", "\\gt"], tr = [0, 1.2, 1.8, 2.4, 3], fc = function(e, t, r, a, i) {
  if (e === "<" || e === "\\lt" || e === "⟨" ? e = "\\langle" : (e === ">" || e === "\\gt" || e === "⟩") && (e = "\\rangle"), Z.contains(Ks, e) || Z.contains(Qs, e))
    return Xs(e, t, !1, r, a, i);
  if (Z.contains(mc, e))
    return Zs(e, tr[t], !1, r, a, i);
  throw new L("Illegal delimiter: '" + e + "'");
}, pc = [{
  type: "small",
  style: Q.SCRIPTSCRIPT
}, {
  type: "small",
  style: Q.SCRIPT
}, {
  type: "small",
  style: Q.TEXT
}, {
  type: "large",
  size: 1
}, {
  type: "large",
  size: 2
}, {
  type: "large",
  size: 3
}, {
  type: "large",
  size: 4
}], gc = [{
  type: "small",
  style: Q.SCRIPTSCRIPT
}, {
  type: "small",
  style: Q.SCRIPT
}, {
  type: "small",
  style: Q.TEXT
}, {
  type: "stack"
}], Js = [{
  type: "small",
  style: Q.SCRIPTSCRIPT
}, {
  type: "small",
  style: Q.SCRIPT
}, {
  type: "small",
  style: Q.TEXT
}, {
  type: "large",
  size: 1
}, {
  type: "large",
  size: 2
}, {
  type: "large",
  size: 3
}, {
  type: "large",
  size: 4
}, {
  type: "stack"
}], vc = function(e) {
  if (e.type === "small")
    return "Main-Regular";
  if (e.type === "large")
    return "Size" + e.size + "-Regular";
  if (e.type === "stack")
    return "Size4-Regular";
  throw new Error("Add support for delim type '" + e.type + "' here.");
}, eo = function(e, t, r, a) {
  for (var i = Math.min(2, 3 - a.style.size), l = i; l < r.length && r[l].type !== "stack"; l++) {
    var s = K0(e, vc(r[l]), "math"), u = s.height + s.depth;
    if (r[l].type === "small") {
      var h = a.havingBaseStyle(r[l].style);
      u *= h.sizeMultiplier;
    }
    if (u > t)
      return r[l];
  }
  return r[r.length - 1];
}, to = function(e, t, r, a, i, l) {
  e === "<" || e === "\\lt" || e === "⟨" ? e = "\\langle" : (e === ">" || e === "\\gt" || e === "⟩") && (e = "\\rangle");
  var s;
  Z.contains(Qs, e) ? s = pc : Z.contains(Ks, e) ? s = Js : s = gc;
  var u = eo(e, t, s, a);
  return u.type === "small" ? oc(e, u.style, r, a, i, l) : u.type === "large" ? Xs(e, u.size, r, a, i, l) : Zs(e, t, r, a, i, l);
}, _c = function(e, t, r, a, i, l) {
  var s = a.fontMetrics().axisHeight * a.sizeMultiplier, u = 901, h = 5 / a.fontMetrics().ptPerEm, d = Math.max(t - s, r + s), g = Math.max(
    // In real TeX, calculations are done using integral values which are
    // 65536 per pt, or 655360 per em. So, the division here truncates in
    // TeX but doesn't here, producing different results. If we wanted to
    // exactly match TeX's calculation, we could do
    //   Math.floor(655360 * maxDistFromAxis / 500) *
    //    delimiterFactor / 655360
    // (To see the difference, compare
    //    x^{x^{\left(\rule{0.1em}{0.68em}\right)}}
    // in TeX and KaTeX)
    d / 500 * u,
    2 * d - h
  );
  return to(e, g, !0, a, i, l);
}, t0 = {
  sqrtImage: dc,
  sizedDelim: fc,
  sizeToMaxHeight: tr,
  customSizedDelim: to,
  leftRightDelim: _c
}, Ii = {
  "\\bigl": {
    mclass: "mopen",
    size: 1
  },
  "\\Bigl": {
    mclass: "mopen",
    size: 2
  },
  "\\biggl": {
    mclass: "mopen",
    size: 3
  },
  "\\Biggl": {
    mclass: "mopen",
    size: 4
  },
  "\\bigr": {
    mclass: "mclose",
    size: 1
  },
  "\\Bigr": {
    mclass: "mclose",
    size: 2
  },
  "\\biggr": {
    mclass: "mclose",
    size: 3
  },
  "\\Biggr": {
    mclass: "mclose",
    size: 4
  },
  "\\bigm": {
    mclass: "mrel",
    size: 1
  },
  "\\Bigm": {
    mclass: "mrel",
    size: 2
  },
  "\\biggm": {
    mclass: "mrel",
    size: 3
  },
  "\\Biggm": {
    mclass: "mrel",
    size: 4
  },
  "\\big": {
    mclass: "mord",
    size: 1
  },
  "\\Big": {
    mclass: "mord",
    size: 2
  },
  "\\bigg": {
    mclass: "mord",
    size: 3
  },
  "\\Bigg": {
    mclass: "mord",
    size: 4
  }
}, bc = ["(", "\\lparen", ")", "\\rparen", "[", "\\lbrack", "]", "\\rbrack", "\\{", "\\lbrace", "\\}", "\\rbrace", "\\lfloor", "\\rfloor", "⌊", "⌋", "\\lceil", "\\rceil", "⌈", "⌉", "<", ">", "\\langle", "⟨", "\\rangle", "⟩", "\\lt", "\\gt", "\\lvert", "\\rvert", "\\lVert", "\\rVert", "\\lgroup", "\\rgroup", "⟮", "⟯", "\\lmoustache", "\\rmoustache", "⎰", "⎱", "/", "\\backslash", "|", "\\vert", "\\|", "\\Vert", "\\uparrow", "\\Uparrow", "\\downarrow", "\\Downarrow", "\\updownarrow", "\\Updownarrow", "."];
function cn(n, e) {
  var t = on(n);
  if (t && Z.contains(bc, t.text))
    return t;
  throw t ? new L("Invalid delimiter '" + t.text + "' after '" + e.funcName + "'", n) : new L("Invalid delimiter type '" + n.type + "'", n);
}
H({
  type: "delimsizing",
  names: ["\\bigl", "\\Bigl", "\\biggl", "\\Biggl", "\\bigr", "\\Bigr", "\\biggr", "\\Biggr", "\\bigm", "\\Bigm", "\\biggm", "\\Biggm", "\\big", "\\Big", "\\bigg", "\\Bigg"],
  props: {
    numArgs: 1,
    argTypes: ["primitive"]
  },
  handler: (n, e) => {
    var t = cn(e[0], n);
    return {
      type: "delimsizing",
      mode: n.parser.mode,
      size: Ii[n.funcName].size,
      mclass: Ii[n.funcName].mclass,
      delim: t.text
    };
  },
  htmlBuilder: (n, e) => n.delim === "." ? F.makeSpan([n.mclass]) : t0.sizedDelim(n.delim, n.size, e, n.mode, [n.mclass]),
  mathmlBuilder: (n) => {
    var e = [];
    n.delim !== "." && e.push(ct(n.delim, n.mode));
    var t = new q.MathNode("mo", e);
    n.mclass === "mopen" || n.mclass === "mclose" ? t.setAttribute("fence", "true") : t.setAttribute("fence", "false"), t.setAttribute("stretchy", "true");
    var r = P(t0.sizeToMaxHeight[n.size]);
    return t.setAttribute("minsize", r), t.setAttribute("maxsize", r), t;
  }
});
function Oi(n) {
  if (!n.body)
    throw new Error("Bug: The leftright ParseNode wasn't fully parsed.");
}
H({
  type: "leftright-right",
  names: ["\\right"],
  props: {
    numArgs: 1,
    primitive: !0
  },
  handler: (n, e) => {
    var t = n.parser.gullet.macros.get("\\current@color");
    if (t && typeof t != "string")
      throw new L("\\current@color set to non-string in \\right");
    return {
      type: "leftright-right",
      mode: n.parser.mode,
      delim: cn(e[0], n).text,
      color: t
      // undefined if not set via \color
    };
  }
});
H({
  type: "leftright",
  names: ["\\left"],
  props: {
    numArgs: 1,
    primitive: !0
  },
  handler: (n, e) => {
    var t = cn(e[0], n), r = n.parser;
    ++r.leftrightDepth;
    var a = r.parseExpression(!1);
    --r.leftrightDepth, r.expect("\\right", !1);
    var i = re(r.parseFunction(), "leftright-right");
    return {
      type: "leftright",
      mode: r.mode,
      body: a,
      left: t.text,
      right: i.delim,
      rightColor: i.color
    };
  },
  htmlBuilder: (n, e) => {
    Oi(n);
    for (var t = Me(n.body, e, !0, ["mopen", "mclose"]), r = 0, a = 0, i = !1, l = 0; l < t.length; l++)
      t[l].isMiddle ? i = !0 : (r = Math.max(t[l].height, r), a = Math.max(t[l].depth, a));
    r *= e.sizeMultiplier, a *= e.sizeMultiplier;
    var s;
    if (n.left === "." ? s = or(e, ["mopen"]) : s = t0.leftRightDelim(n.left, r, a, e, n.mode, ["mopen"]), t.unshift(s), i)
      for (var u = 1; u < t.length; u++) {
        var h = t[u], d = h.isMiddle;
        d && (t[u] = t0.leftRightDelim(d.delim, r, a, d.options, n.mode, []));
      }
    var g;
    if (n.right === ".")
      g = or(e, ["mclose"]);
    else {
      var p = n.rightColor ? e.withColor(n.rightColor) : e;
      g = t0.leftRightDelim(n.right, r, a, p, n.mode, ["mclose"]);
    }
    return t.push(g), F.makeSpan(["minner"], t, e);
  },
  mathmlBuilder: (n, e) => {
    Oi(n);
    var t = Ze(n.body, e);
    if (n.left !== ".") {
      var r = new q.MathNode("mo", [ct(n.left, n.mode)]);
      r.setAttribute("fence", "true"), t.unshift(r);
    }
    if (n.right !== ".") {
      var a = new q.MathNode("mo", [ct(n.right, n.mode)]);
      a.setAttribute("fence", "true"), n.rightColor && a.setAttribute("mathcolor", n.rightColor), t.push(a);
    }
    return Fa(t);
  }
});
H({
  type: "middle",
  names: ["\\middle"],
  props: {
    numArgs: 1,
    primitive: !0
  },
  handler: (n, e) => {
    var t = cn(e[0], n);
    if (!n.parser.leftrightDepth)
      throw new L("\\middle without preceding \\left", t);
    return {
      type: "middle",
      mode: n.parser.mode,
      delim: t.text
    };
  },
  htmlBuilder: (n, e) => {
    var t;
    if (n.delim === ".")
      t = or(e, []);
    else {
      t = t0.sizedDelim(n.delim, 1, e, n.mode, []);
      var r = {
        delim: n.delim,
        options: e
      };
      t.isMiddle = r;
    }
    return t;
  },
  mathmlBuilder: (n, e) => {
    var t = n.delim === "\\vert" || n.delim === "|" ? ct("|", "text") : ct(n.delim, n.mode), r = new q.MathNode("mo", [t]);
    return r.setAttribute("fence", "true"), r.setAttribute("lspace", "0.05em"), r.setAttribute("rspace", "0.05em"), r;
  }
});
var za = (n, e) => {
  var t = F.wrapFragment(le(n.body, e), e), r = n.label.slice(1), a = e.sizeMultiplier, i, l = 0, s = Z.isCharacterBox(n.body);
  if (r === "sout")
    i = F.makeSpan(["stretchy", "sout"]), i.height = e.fontMetrics().defaultRuleThickness / a, l = -0.5 * e.fontMetrics().xHeight;
  else if (r === "phase") {
    var u = ke({
      number: 0.6,
      unit: "pt"
    }, e), h = ke({
      number: 0.35,
      unit: "ex"
    }, e), d = e.havingBaseSizing();
    a = a / d.sizeMultiplier;
    var g = t.height + t.depth + u + h;
    t.style.paddingLeft = P(g / 2 + u);
    var p = Math.floor(1e3 * g * a), v = d1(p), k = new r0([new m0("phase", v)], {
      width: "400em",
      height: P(p / 1e3),
      viewBox: "0 0 400000 " + p,
      preserveAspectRatio: "xMinYMin slice"
    });
    i = F.makeSvgSpan(["hide-tail"], [k], e), i.style.height = P(g), l = t.depth + u + h;
  } else {
    /cancel/.test(r) ? s || t.classes.push("cancel-pad") : r === "angl" ? t.classes.push("anglpad") : t.classes.push("boxpad");
    var A = 0, C = 0, z = 0;
    /box/.test(r) ? (z = Math.max(
      e.fontMetrics().fboxrule,
      // default
      e.minRuleThickness
      // User override.
    ), A = e.fontMetrics().fboxsep + (r === "colorbox" ? 0 : z), C = A) : r === "angl" ? (z = Math.max(e.fontMetrics().defaultRuleThickness, e.minRuleThickness), A = 4 * z, C = Math.max(0, 0.25 - t.depth)) : (A = s ? 0.2 : 0, C = A), i = a0.encloseSpan(t, r, A, C, e), /fbox|boxed|fcolorbox/.test(r) ? (i.style.borderStyle = "solid", i.style.borderWidth = P(z)) : r === "angl" && z !== 0.049 && (i.style.borderTopWidth = P(z), i.style.borderRightWidth = P(z)), l = t.depth + C, n.backgroundColor && (i.style.backgroundColor = n.backgroundColor, n.borderColor && (i.style.borderColor = n.borderColor));
  }
  var x;
  if (n.backgroundColor)
    x = F.makeVList({
      positionType: "individualShift",
      children: [
        // Put the color background behind inner;
        {
          type: "elem",
          elem: i,
          shift: l
        },
        {
          type: "elem",
          elem: t,
          shift: 0
        }
      ]
    }, e);
  else {
    var _ = /cancel|phase/.test(r) ? ["svg-align"] : [];
    x = F.makeVList({
      positionType: "individualShift",
      children: [
        // Write the \cancel stroke on top of inner.
        {
          type: "elem",
          elem: t,
          shift: 0
        },
        {
          type: "elem",
          elem: i,
          shift: l,
          wrapperClasses: _
        }
      ]
    }, e);
  }
  return /cancel/.test(r) && (x.height = t.height, x.depth = t.depth), /cancel/.test(r) && !s ? F.makeSpan(["mord", "cancel-lap"], [x], e) : F.makeSpan(["mord"], [x], e);
}, Ba = (n, e) => {
  var t = 0, r = new q.MathNode(n.label.indexOf("colorbox") > -1 ? "mpadded" : "menclose", [pe(n.body, e)]);
  switch (n.label) {
    case "\\cancel":
      r.setAttribute("notation", "updiagonalstrike");
      break;
    case "\\bcancel":
      r.setAttribute("notation", "downdiagonalstrike");
      break;
    case "\\phase":
      r.setAttribute("notation", "phasorangle");
      break;
    case "\\sout":
      r.setAttribute("notation", "horizontalstrike");
      break;
    case "\\fbox":
      r.setAttribute("notation", "box");
      break;
    case "\\angl":
      r.setAttribute("notation", "actuarial");
      break;
    case "\\fcolorbox":
    case "\\colorbox":
      if (t = e.fontMetrics().fboxsep * e.fontMetrics().ptPerEm, r.setAttribute("width", "+" + 2 * t + "pt"), r.setAttribute("height", "+" + 2 * t + "pt"), r.setAttribute("lspace", t + "pt"), r.setAttribute("voffset", t + "pt"), n.label === "\\fcolorbox") {
        var a = Math.max(
          e.fontMetrics().fboxrule,
          // default
          e.minRuleThickness
          // user override
        );
        r.setAttribute("style", "border: " + a + "em solid " + String(n.borderColor));
      }
      break;
    case "\\xcancel":
      r.setAttribute("notation", "updiagonalstrike downdiagonalstrike");
      break;
  }
  return n.backgroundColor && r.setAttribute("mathbackground", n.backgroundColor), r;
};
H({
  type: "enclose",
  names: ["\\colorbox"],
  props: {
    numArgs: 2,
    allowedInText: !0,
    argTypes: ["color", "text"]
  },
  handler(n, e, t) {
    var {
      parser: r,
      funcName: a
    } = n, i = re(e[0], "color-token").color, l = e[1];
    return {
      type: "enclose",
      mode: r.mode,
      label: a,
      backgroundColor: i,
      body: l
    };
  },
  htmlBuilder: za,
  mathmlBuilder: Ba
});
H({
  type: "enclose",
  names: ["\\fcolorbox"],
  props: {
    numArgs: 3,
    allowedInText: !0,
    argTypes: ["color", "color", "text"]
  },
  handler(n, e, t) {
    var {
      parser: r,
      funcName: a
    } = n, i = re(e[0], "color-token").color, l = re(e[1], "color-token").color, s = e[2];
    return {
      type: "enclose",
      mode: r.mode,
      label: a,
      backgroundColor: l,
      borderColor: i,
      body: s
    };
  },
  htmlBuilder: za,
  mathmlBuilder: Ba
});
H({
  type: "enclose",
  names: ["\\fbox"],
  props: {
    numArgs: 1,
    argTypes: ["hbox"],
    allowedInText: !0
  },
  handler(n, e) {
    var {
      parser: t
    } = n;
    return {
      type: "enclose",
      mode: t.mode,
      label: "\\fbox",
      body: e[0]
    };
  }
});
H({
  type: "enclose",
  names: ["\\cancel", "\\bcancel", "\\xcancel", "\\sout", "\\phase"],
  props: {
    numArgs: 1
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0];
    return {
      type: "enclose",
      mode: t.mode,
      label: r,
      body: a
    };
  },
  htmlBuilder: za,
  mathmlBuilder: Ba
});
H({
  type: "enclose",
  names: ["\\angl"],
  props: {
    numArgs: 1,
    argTypes: ["hbox"],
    allowedInText: !1
  },
  handler(n, e) {
    var {
      parser: t
    } = n;
    return {
      type: "enclose",
      mode: t.mode,
      label: "\\angl",
      body: e[0]
    };
  }
});
var ro = {};
function Pt(n) {
  for (var {
    type: e,
    names: t,
    props: r,
    handler: a,
    htmlBuilder: i,
    mathmlBuilder: l
  } = n, s = {
    type: e,
    numArgs: r.numArgs || 0,
    allowedInText: !1,
    numOptionalArgs: 0,
    handler: a
  }, u = 0; u < t.length; ++u)
    ro[t[u]] = s;
  i && (Qr[e] = i), l && (Jr[e] = l);
}
var no = {};
function f(n, e) {
  no[n] = e;
}
function Pi(n) {
  var e = [];
  n.consumeSpaces();
  var t = n.fetch().text;
  for (t === "\\relax" && (n.consume(), n.consumeSpaces(), t = n.fetch().text); t === "\\hline" || t === "\\hdashline"; )
    n.consume(), e.push(t === "\\hdashline"), n.consumeSpaces(), t = n.fetch().text;
  return e;
}
var hn = (n) => {
  var e = n.parser.settings;
  if (!e.displayMode)
    throw new L("{" + n.envName + "} can be used only in display mode.");
};
function Ra(n) {
  if (n.indexOf("ed") === -1)
    return n.indexOf("*") === -1;
}
function p0(n, e, t) {
  var {
    hskipBeforeAndAfter: r,
    addJot: a,
    cols: i,
    arraystretch: l,
    colSeparationType: s,
    autoTag: u,
    singleRow: h,
    emptySingleRow: d,
    maxNumCols: g,
    leqno: p
  } = e;
  if (n.gullet.beginGroup(), h || n.gullet.macros.set("\\cr", "\\\\\\relax"), !l) {
    var v = n.gullet.expandMacroAsText("\\arraystretch");
    if (v == null)
      l = 1;
    else if (l = parseFloat(v), !l || l < 0)
      throw new L("Invalid \\arraystretch: " + v);
  }
  n.gullet.beginGroup();
  var k = [], A = [k], C = [], z = [], x = u != null ? [] : void 0;
  function _() {
    u && n.gullet.macros.set("\\@eqnsw", "1", !0);
  }
  function w() {
    x && (n.gullet.macros.get("\\df@tag") ? (x.push(n.subparse([new ot("\\df@tag")])), n.gullet.macros.set("\\df@tag", void 0, !0)) : x.push(!!u && n.gullet.macros.get("\\@eqnsw") === "1"));
  }
  for (_(), z.push(Pi(n)); ; ) {
    var E = n.parseExpression(!1, h ? "\\end" : "\\\\");
    n.gullet.endGroup(), n.gullet.beginGroup(), E = {
      type: "ordgroup",
      mode: n.mode,
      body: E
    }, t && (E = {
      type: "styling",
      mode: n.mode,
      style: t,
      body: [E]
    }), k.push(E);
    var T = n.fetch().text;
    if (T === "&") {
      if (g && k.length === g) {
        if (h || s)
          throw new L("Too many tab characters: &", n.nextToken);
        n.settings.reportNonstrict("textEnv", "Too few columns specified in the {array} column argument.");
      }
      n.consume();
    } else if (T === "\\end") {
      w(), k.length === 1 && E.type === "styling" && E.body[0].body.length === 0 && (A.length > 1 || !d) && A.pop(), z.length < A.length + 1 && z.push([]);
      break;
    } else if (T === "\\\\") {
      n.consume();
      var $ = void 0;
      n.gullet.future().text !== " " && ($ = n.parseSizeGroup(!0)), C.push($ ? $.value : null), w(), z.push(Pi(n)), k = [], A.push(k), _();
    } else
      throw new L("Expected & or \\\\ or \\cr or \\end", n.nextToken);
  }
  return n.gullet.endGroup(), n.gullet.endGroup(), {
    type: "array",
    mode: n.mode,
    addJot: a,
    arraystretch: l,
    body: A,
    cols: i,
    rowGaps: C,
    hskipBeforeAndAfter: r,
    hLinesBeforeRow: z,
    colSeparationType: s,
    tags: x,
    leqno: p
  };
}
function Na(n) {
  return n.slice(0, 1) === "d" ? "display" : "text";
}
var Ht = function(e, t) {
  var r, a, i = e.body.length, l = e.hLinesBeforeRow, s = 0, u = new Array(i), h = [], d = Math.max(
    // From LaTeX \showthe\arrayrulewidth. Equals 0.04 em.
    t.fontMetrics().arrayRuleWidth,
    t.minRuleThickness
    // User override.
  ), g = 1 / t.fontMetrics().ptPerEm, p = 5 * g;
  if (e.colSeparationType && e.colSeparationType === "small") {
    var v = t.havingStyle(Q.SCRIPT).sizeMultiplier;
    p = 0.2778 * (v / t.sizeMultiplier);
  }
  var k = e.colSeparationType === "CD" ? ke({
    number: 3,
    unit: "ex"
  }, t) : 12 * g, A = 3 * g, C = e.arraystretch * k, z = 0.7 * C, x = 0.3 * C, _ = 0;
  function w(Vt) {
    for (var Wt = 0; Wt < Vt.length; ++Wt)
      Wt > 0 && (_ += 0.25), h.push({
        pos: _,
        isDashed: Vt[Wt]
      });
  }
  for (w(l[0]), r = 0; r < e.body.length; ++r) {
    var E = e.body[r], T = z, $ = x;
    s < E.length && (s = E.length);
    var M = new Array(E.length);
    for (a = 0; a < E.length; ++a) {
      var B = le(E[a], t);
      $ < B.depth && ($ = B.depth), T < B.height && (T = B.height), M[a] = B;
    }
    var G = e.rowGaps[r], U = 0;
    G && (U = ke(G, t), U > 0 && (U += x, $ < U && ($ = U), U = 0)), e.addJot && ($ += A), M.height = T, M.depth = $, _ += T, M.pos = _, _ += $ + U, u[r] = M, w(l[r + 1]);
  }
  var j = _ / 2 + t.fontMetrics().axisHeight, oe = e.cols || [], ee = [], ue, fe, Ee = [];
  if (e.tags && e.tags.some((Vt) => Vt))
    for (r = 0; r < i; ++r) {
      var ne = u[r], ve = ne.pos - j, we = e.tags[r], N = void 0;
      we === !0 ? N = F.makeSpan(["eqn-num"], [], t) : we === !1 ? N = F.makeSpan([], [], t) : N = F.makeSpan([], Me(we, t, !0), t), N.depth = ne.depth, N.height = ne.height, Ee.push({
        type: "elem",
        elem: N,
        shift: ve
      });
    }
  for (
    a = 0, fe = 0;
    // Continue while either there are more columns or more column
    // descriptions, so trailing separators don't get lost.
    a < s || fe < oe.length;
    ++a, ++fe
  ) {
    for (var se = oe[fe] || {}, ce = !0; se.type === "separator"; ) {
      if (ce || (ue = F.makeSpan(["arraycolsep"], []), ue.style.width = P(t.fontMetrics().doubleRuleSep), ee.push(ue)), se.separator === "|" || se.separator === ":") {
        var Ce = se.separator === "|" ? "solid" : "dashed", O = F.makeSpan(["vertical-separator"], [], t);
        O.style.height = P(_), O.style.borderRightWidth = P(d), O.style.borderRightStyle = Ce, O.style.margin = "0 " + P(-d / 2);
        var Ie = _ - j;
        Ie && (O.style.verticalAlign = P(-Ie)), ee.push(O);
      } else
        throw new L("Invalid separator type: " + se.separator);
      fe++, se = oe[fe] || {}, ce = !1;
    }
    if (!(a >= s)) {
      var Oe = void 0;
      (a > 0 || e.hskipBeforeAndAfter) && (Oe = Z.deflt(se.pregap, p), Oe !== 0 && (ue = F.makeSpan(["arraycolsep"], []), ue.style.width = P(Oe), ee.push(ue)));
      var Ke = [];
      for (r = 0; r < i; ++r) {
        var ft = u[r], pt = ft[a];
        if (pt) {
          var Gt = ft.pos - j;
          pt.depth = ft.depth, pt.height = ft.height, Ke.push({
            type: "elem",
            elem: pt,
            shift: Gt
          });
        }
      }
      Ke = F.makeVList({
        positionType: "individualShift",
        children: Ke
      }, t), Ke = F.makeSpan(["col-align-" + (se.align || "c")], [Ke]), ee.push(Ke), (a < s - 1 || e.hskipBeforeAndAfter) && (Oe = Z.deflt(se.postgap, p), Oe !== 0 && (ue = F.makeSpan(["arraycolsep"], []), ue.style.width = P(Oe), ee.push(ue)));
    }
  }
  if (u = F.makeSpan(["mtable"], ee), h.length > 0) {
    for (var At = F.makeLineSpan("hline", t, d), Et = F.makeLineSpan("hdashline", t, d), gt = [{
      type: "elem",
      elem: u,
      shift: 0
    }]; h.length > 0; ) {
      var D0 = h.pop(), S0 = D0.pos - j;
      D0.isDashed ? gt.push({
        type: "elem",
        elem: Et,
        shift: S0
      }) : gt.push({
        type: "elem",
        elem: At,
        shift: S0
      });
    }
    u = F.makeVList({
      positionType: "individualShift",
      children: gt
    }, t);
  }
  if (Ee.length === 0)
    return F.makeSpan(["mord"], [u], t);
  var Ft = F.makeVList({
    positionType: "individualShift",
    children: Ee
  }, t);
  return Ft = F.makeSpan(["tag"], [Ft], t), F.makeFragment([u, Ft]);
}, yc = {
  c: "center ",
  l: "left ",
  r: "right "
}, Ut = function(e, t) {
  for (var r = [], a = new q.MathNode("mtd", [], ["mtr-glue"]), i = new q.MathNode("mtd", [], ["mml-eqn-num"]), l = 0; l < e.body.length; l++) {
    for (var s = e.body[l], u = [], h = 0; h < s.length; h++)
      u.push(new q.MathNode("mtd", [pe(s[h], t)]));
    e.tags && e.tags[l] && (u.unshift(a), u.push(a), e.leqno ? u.unshift(i) : u.push(i)), r.push(new q.MathNode("mtr", u));
  }
  var d = new q.MathNode("mtable", r), g = e.arraystretch === 0.5 ? 0.1 : 0.16 + e.arraystretch - 1 + (e.addJot ? 0.09 : 0);
  d.setAttribute("rowspacing", P(g));
  var p = "", v = "";
  if (e.cols && e.cols.length > 0) {
    var k = e.cols, A = "", C = !1, z = 0, x = k.length;
    k[0].type === "separator" && (p += "top ", z = 1), k[k.length - 1].type === "separator" && (p += "bottom ", x -= 1);
    for (var _ = z; _ < x; _++)
      k[_].type === "align" ? (v += yc[k[_].align], C && (A += "none "), C = !0) : k[_].type === "separator" && C && (A += k[_].separator === "|" ? "solid " : "dashed ", C = !1);
    d.setAttribute("columnalign", v.trim()), /[sd]/.test(A) && d.setAttribute("columnlines", A.trim());
  }
  if (e.colSeparationType === "align") {
    for (var w = e.cols || [], E = "", T = 1; T < w.length; T++)
      E += T % 2 ? "0em " : "1em ";
    d.setAttribute("columnspacing", E.trim());
  } else e.colSeparationType === "alignat" || e.colSeparationType === "gather" ? d.setAttribute("columnspacing", "0em") : e.colSeparationType === "small" ? d.setAttribute("columnspacing", "0.2778em") : e.colSeparationType === "CD" ? d.setAttribute("columnspacing", "0.5em") : d.setAttribute("columnspacing", "1em");
  var $ = "", M = e.hLinesBeforeRow;
  p += M[0].length > 0 ? "left " : "", p += M[M.length - 1].length > 0 ? "right " : "";
  for (var B = 1; B < M.length - 1; B++)
    $ += M[B].length === 0 ? "none " : M[B][0] ? "dashed " : "solid ";
  return /[sd]/.test($) && d.setAttribute("rowlines", $.trim()), p !== "" && (d = new q.MathNode("menclose", [d]), d.setAttribute("notation", p.trim())), e.arraystretch && e.arraystretch < 1 && (d = new q.MathNode("mstyle", [d]), d.setAttribute("scriptlevel", "1")), d;
}, ao = function(e, t) {
  e.envName.indexOf("ed") === -1 && hn(e);
  var r = [], a = e.envName.indexOf("at") > -1 ? "alignat" : "align", i = e.envName === "split", l = p0(e.parser, {
    cols: r,
    addJot: !0,
    autoTag: i ? void 0 : Ra(e.envName),
    emptySingleRow: !0,
    colSeparationType: a,
    maxNumCols: i ? 2 : void 0,
    leqno: e.parser.settings.leqno
  }, "display"), s, u = 0, h = {
    type: "ordgroup",
    mode: e.mode,
    body: []
  };
  if (t[0] && t[0].type === "ordgroup") {
    for (var d = "", g = 0; g < t[0].body.length; g++) {
      var p = re(t[0].body[g], "textord");
      d += p.text;
    }
    s = Number(d), u = s * 2;
  }
  var v = !u;
  l.body.forEach(function(z) {
    for (var x = 1; x < z.length; x += 2) {
      var _ = re(z[x], "styling"), w = re(_.body[0], "ordgroup");
      w.body.unshift(h);
    }
    if (v)
      u < z.length && (u = z.length);
    else {
      var E = z.length / 2;
      if (s < E)
        throw new L("Too many math in a row: " + ("expected " + s + ", but got " + E), z[0]);
    }
  });
  for (var k = 0; k < u; ++k) {
    var A = "r", C = 0;
    k % 2 === 1 ? A = "l" : k > 0 && v && (C = 1), r[k] = {
      type: "align",
      align: A,
      pregap: C,
      postgap: 0
    };
  }
  return l.colSeparationType = v ? "align" : "alignat", l;
};
Pt({
  type: "array",
  names: ["array", "darray"],
  props: {
    numArgs: 1
  },
  handler(n, e) {
    var t = on(e[0]), r = t ? [e[0]] : re(e[0], "ordgroup").body, a = r.map(function(l) {
      var s = Ta(l), u = s.text;
      if ("lcr".indexOf(u) !== -1)
        return {
          type: "align",
          align: u
        };
      if (u === "|")
        return {
          type: "separator",
          separator: "|"
        };
      if (u === ":")
        return {
          type: "separator",
          separator: ":"
        };
      throw new L("Unknown column alignment: " + u, l);
    }), i = {
      cols: a,
      hskipBeforeAndAfter: !0,
      // \@preamble in lttab.dtx
      maxNumCols: a.length
    };
    return p0(n.parser, i, Na(n.envName));
  },
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "matrix*", "pmatrix*", "bmatrix*", "Bmatrix*", "vmatrix*", "Vmatrix*"],
  props: {
    numArgs: 0
  },
  handler(n) {
    var e = {
      matrix: null,
      pmatrix: ["(", ")"],
      bmatrix: ["[", "]"],
      Bmatrix: ["\\{", "\\}"],
      vmatrix: ["|", "|"],
      Vmatrix: ["\\Vert", "\\Vert"]
    }[n.envName.replace("*", "")], t = "c", r = {
      hskipBeforeAndAfter: !1,
      cols: [{
        type: "align",
        align: t
      }]
    };
    if (n.envName.charAt(n.envName.length - 1) === "*") {
      var a = n.parser;
      if (a.consumeSpaces(), a.fetch().text === "[") {
        if (a.consume(), a.consumeSpaces(), t = a.fetch().text, "lcr".indexOf(t) === -1)
          throw new L("Expected l or c or r", a.nextToken);
        a.consume(), a.consumeSpaces(), a.expect("]"), a.consume(), r.cols = [{
          type: "align",
          align: t
        }];
      }
    }
    var i = p0(n.parser, r, Na(n.envName)), l = Math.max(0, ...i.body.map((s) => s.length));
    return i.cols = new Array(l).fill({
      type: "align",
      align: t
    }), e ? {
      type: "leftright",
      mode: n.mode,
      body: [i],
      left: e[0],
      right: e[1],
      rightColor: void 0
      // \right uninfluenced by \color in array
    } : i;
  },
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["smallmatrix"],
  props: {
    numArgs: 0
  },
  handler(n) {
    var e = {
      arraystretch: 0.5
    }, t = p0(n.parser, e, "script");
    return t.colSeparationType = "small", t;
  },
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["subarray"],
  props: {
    numArgs: 1
  },
  handler(n, e) {
    var t = on(e[0]), r = t ? [e[0]] : re(e[0], "ordgroup").body, a = r.map(function(l) {
      var s = Ta(l), u = s.text;
      if ("lc".indexOf(u) !== -1)
        return {
          type: "align",
          align: u
        };
      throw new L("Unknown column alignment: " + u, l);
    });
    if (a.length > 1)
      throw new L("{subarray} can contain only one column");
    var i = {
      cols: a,
      hskipBeforeAndAfter: !1,
      arraystretch: 0.5
    };
    if (i = p0(n.parser, i, "script"), i.body.length > 0 && i.body[0].length > 1)
      throw new L("{subarray} can contain only one column");
    return i;
  },
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["cases", "dcases", "rcases", "drcases"],
  props: {
    numArgs: 0
  },
  handler(n) {
    var e = {
      arraystretch: 1.2,
      cols: [{
        type: "align",
        align: "l",
        pregap: 0,
        // TODO(kevinb) get the current style.
        // For now we use the metrics for TEXT style which is what we were
        // doing before.  Before attempting to get the current style we
        // should look at TeX's behavior especially for \over and matrices.
        postgap: 1
        /* 1em quad */
      }, {
        type: "align",
        align: "l",
        pregap: 0,
        postgap: 0
      }]
    }, t = p0(n.parser, e, Na(n.envName));
    return {
      type: "leftright",
      mode: n.mode,
      body: [t],
      left: n.envName.indexOf("r") > -1 ? "." : "\\{",
      right: n.envName.indexOf("r") > -1 ? "\\}" : ".",
      rightColor: void 0
    };
  },
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["align", "align*", "aligned", "split"],
  props: {
    numArgs: 0
  },
  handler: ao,
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["gathered", "gather", "gather*"],
  props: {
    numArgs: 0
  },
  handler(n) {
    Z.contains(["gather", "gather*"], n.envName) && hn(n);
    var e = {
      cols: [{
        type: "align",
        align: "c"
      }],
      addJot: !0,
      colSeparationType: "gather",
      autoTag: Ra(n.envName),
      emptySingleRow: !0,
      leqno: n.parser.settings.leqno
    };
    return p0(n.parser, e, "display");
  },
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["alignat", "alignat*", "alignedat"],
  props: {
    numArgs: 1
  },
  handler: ao,
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["equation", "equation*"],
  props: {
    numArgs: 0
  },
  handler(n) {
    hn(n);
    var e = {
      autoTag: Ra(n.envName),
      emptySingleRow: !0,
      singleRow: !0,
      maxNumCols: 1,
      leqno: n.parser.settings.leqno
    };
    return p0(n.parser, e, "display");
  },
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
Pt({
  type: "array",
  names: ["CD"],
  props: {
    numArgs: 0
  },
  handler(n) {
    return hn(n), lc(n.parser);
  },
  htmlBuilder: Ht,
  mathmlBuilder: Ut
});
f("\\nonumber", "\\gdef\\@eqnsw{0}");
f("\\notag", "\\nonumber");
H({
  type: "text",
  // Doesn't matter what this is.
  names: ["\\hline", "\\hdashline"],
  props: {
    numArgs: 0,
    allowedInText: !0,
    allowedInMath: !0
  },
  handler(n, e) {
    throw new L(n.funcName + " valid only within array environment");
  }
});
var Hi = ro;
H({
  type: "environment",
  names: ["\\begin", "\\end"],
  props: {
    numArgs: 1,
    argTypes: ["text"]
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0];
    if (a.type !== "ordgroup")
      throw new L("Invalid environment name", a);
    for (var i = "", l = 0; l < a.body.length; ++l)
      i += re(a.body[l], "textord").text;
    if (r === "\\begin") {
      if (!Hi.hasOwnProperty(i))
        throw new L("No such environment: " + i, a);
      var s = Hi[i], {
        args: u,
        optArgs: h
      } = t.parseArguments("\\begin{" + i + "}", s), d = {
        mode: t.mode,
        envName: i,
        parser: t
      }, g = s.handler(d, u, h);
      t.expect("\\end", !1);
      var p = t.nextToken, v = re(t.parseFunction(), "environment");
      if (v.name !== i)
        throw new L("Mismatch: \\begin{" + i + "} matched by \\end{" + v.name + "}", p);
      return g;
    }
    return {
      type: "environment",
      mode: t.mode,
      name: i,
      nameGroup: a
    };
  }
});
var io = (n, e) => {
  var t = n.font, r = e.withFont(t);
  return le(n.body, r);
}, lo = (n, e) => {
  var t = n.font, r = e.withFont(t);
  return pe(n.body, r);
}, Ui = {
  "\\Bbb": "\\mathbb",
  "\\bold": "\\mathbf",
  "\\frak": "\\mathfrak",
  "\\bm": "\\boldsymbol"
};
H({
  type: "font",
  names: [
    // styles, except \boldsymbol defined below
    "\\mathrm",
    "\\mathit",
    "\\mathbf",
    "\\mathnormal",
    "\\mathsfit",
    // families
    "\\mathbb",
    "\\mathcal",
    "\\mathfrak",
    "\\mathscr",
    "\\mathsf",
    "\\mathtt",
    // aliases, except \bm defined below
    "\\Bbb",
    "\\bold",
    "\\frak"
  ],
  props: {
    numArgs: 1,
    allowedInArgument: !0
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r
    } = n, a = en(e[0]), i = r;
    return i in Ui && (i = Ui[i]), {
      type: "font",
      mode: t.mode,
      font: i.slice(1),
      body: a
    };
  },
  htmlBuilder: io,
  mathmlBuilder: lo
});
H({
  type: "mclass",
  names: ["\\boldsymbol", "\\bm"],
  props: {
    numArgs: 1
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n, r = e[0], a = Z.isCharacterBox(r);
    return {
      type: "mclass",
      mode: t.mode,
      mclass: un(r),
      body: [{
        type: "font",
        mode: t.mode,
        font: "boldsymbol",
        body: r
      }],
      isCharacterBox: a
    };
  }
});
H({
  type: "font",
  names: ["\\rm", "\\sf", "\\tt", "\\bf", "\\it", "\\cal"],
  props: {
    numArgs: 0,
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r,
      breakOnTokenText: a
    } = n, {
      mode: i
    } = t, l = t.parseExpression(!0, a), s = "math" + r.slice(1);
    return {
      type: "font",
      mode: i,
      font: s,
      body: {
        type: "ordgroup",
        mode: t.mode,
        body: l
      }
    };
  },
  htmlBuilder: io,
  mathmlBuilder: lo
});
var so = (n, e) => {
  var t = e;
  return n === "display" ? t = t.id >= Q.SCRIPT.id ? t.text() : Q.DISPLAY : n === "text" && t.size === Q.DISPLAY.size ? t = Q.TEXT : n === "script" ? t = Q.SCRIPT : n === "scriptscript" && (t = Q.SCRIPTSCRIPT), t;
}, qa = (n, e) => {
  var t = so(n.size, e.style), r = t.fracNum(), a = t.fracDen(), i;
  i = e.havingStyle(r);
  var l = le(n.numer, i, e);
  if (n.continued) {
    var s = 8.5 / e.fontMetrics().ptPerEm, u = 3.5 / e.fontMetrics().ptPerEm;
    l.height = l.height < s ? s : l.height, l.depth = l.depth < u ? u : l.depth;
  }
  i = e.havingStyle(a);
  var h = le(n.denom, i, e), d, g, p;
  n.hasBarLine ? (n.barSize ? (g = ke(n.barSize, e), d = F.makeLineSpan("frac-line", e, g)) : d = F.makeLineSpan("frac-line", e), g = d.height, p = d.height) : (d = null, g = 0, p = e.fontMetrics().defaultRuleThickness);
  var v, k, A;
  t.size === Q.DISPLAY.size || n.size === "display" ? (v = e.fontMetrics().num1, g > 0 ? k = 3 * p : k = 7 * p, A = e.fontMetrics().denom1) : (g > 0 ? (v = e.fontMetrics().num2, k = p) : (v = e.fontMetrics().num3, k = 3 * p), A = e.fontMetrics().denom2);
  var C;
  if (d) {
    var x = e.fontMetrics().axisHeight;
    v - l.depth - (x + 0.5 * g) < k && (v += k - (v - l.depth - (x + 0.5 * g))), x - 0.5 * g - (h.height - A) < k && (A += k - (x - 0.5 * g - (h.height - A)));
    var _ = -(x - 0.5 * g);
    C = F.makeVList({
      positionType: "individualShift",
      children: [{
        type: "elem",
        elem: h,
        shift: A
      }, {
        type: "elem",
        elem: d,
        shift: _
      }, {
        type: "elem",
        elem: l,
        shift: -v
      }]
    }, e);
  } else {
    var z = v - l.depth - (h.height - A);
    z < k && (v += 0.5 * (k - z), A += 0.5 * (k - z)), C = F.makeVList({
      positionType: "individualShift",
      children: [{
        type: "elem",
        elem: h,
        shift: A
      }, {
        type: "elem",
        elem: l,
        shift: -v
      }]
    }, e);
  }
  i = e.havingStyle(t), C.height *= i.sizeMultiplier / e.sizeMultiplier, C.depth *= i.sizeMultiplier / e.sizeMultiplier;
  var w;
  t.size === Q.DISPLAY.size ? w = e.fontMetrics().delim1 : t.size === Q.SCRIPTSCRIPT.size ? w = e.havingStyle(Q.SCRIPT).fontMetrics().delim2 : w = e.fontMetrics().delim2;
  var E, T;
  return n.leftDelim == null ? E = or(e, ["mopen"]) : E = t0.customSizedDelim(n.leftDelim, w, !0, e.havingStyle(t), n.mode, ["mopen"]), n.continued ? T = F.makeSpan([]) : n.rightDelim == null ? T = or(e, ["mclose"]) : T = t0.customSizedDelim(n.rightDelim, w, !0, e.havingStyle(t), n.mode, ["mclose"]), F.makeSpan(["mord"].concat(i.sizingClasses(e)), [E, F.makeSpan(["mfrac"], [C]), T], e);
}, La = (n, e) => {
  var t = new q.MathNode("mfrac", [pe(n.numer, e), pe(n.denom, e)]);
  if (!n.hasBarLine)
    t.setAttribute("linethickness", "0px");
  else if (n.barSize) {
    var r = ke(n.barSize, e);
    t.setAttribute("linethickness", P(r));
  }
  var a = so(n.size, e.style);
  if (a.size !== e.style.size) {
    t = new q.MathNode("mstyle", [t]);
    var i = a.size === Q.DISPLAY.size ? "true" : "false";
    t.setAttribute("displaystyle", i), t.setAttribute("scriptlevel", "0");
  }
  if (n.leftDelim != null || n.rightDelim != null) {
    var l = [];
    if (n.leftDelim != null) {
      var s = new q.MathNode("mo", [new q.TextNode(n.leftDelim.replace("\\", ""))]);
      s.setAttribute("fence", "true"), l.push(s);
    }
    if (l.push(t), n.rightDelim != null) {
      var u = new q.MathNode("mo", [new q.TextNode(n.rightDelim.replace("\\", ""))]);
      u.setAttribute("fence", "true"), l.push(u);
    }
    return Fa(l);
  }
  return t;
};
H({
  type: "genfrac",
  names: [
    "\\dfrac",
    "\\frac",
    "\\tfrac",
    "\\dbinom",
    "\\binom",
    "\\tbinom",
    "\\\\atopfrac",
    // can’t be entered directly
    "\\\\bracefrac",
    "\\\\brackfrac"
    // ditto
  ],
  props: {
    numArgs: 2,
    allowedInArgument: !0
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0], i = e[1], l, s = null, u = null, h = "auto";
    switch (r) {
      case "\\dfrac":
      case "\\frac":
      case "\\tfrac":
        l = !0;
        break;
      case "\\\\atopfrac":
        l = !1;
        break;
      case "\\dbinom":
      case "\\binom":
      case "\\tbinom":
        l = !1, s = "(", u = ")";
        break;
      case "\\\\bracefrac":
        l = !1, s = "\\{", u = "\\}";
        break;
      case "\\\\brackfrac":
        l = !1, s = "[", u = "]";
        break;
      default:
        throw new Error("Unrecognized genfrac command");
    }
    switch (r) {
      case "\\dfrac":
      case "\\dbinom":
        h = "display";
        break;
      case "\\tfrac":
      case "\\tbinom":
        h = "text";
        break;
    }
    return {
      type: "genfrac",
      mode: t.mode,
      continued: !1,
      numer: a,
      denom: i,
      hasBarLine: l,
      leftDelim: s,
      rightDelim: u,
      size: h,
      barSize: null
    };
  },
  htmlBuilder: qa,
  mathmlBuilder: La
});
H({
  type: "genfrac",
  names: ["\\cfrac"],
  props: {
    numArgs: 2
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0], i = e[1];
    return {
      type: "genfrac",
      mode: t.mode,
      continued: !0,
      numer: a,
      denom: i,
      hasBarLine: !0,
      leftDelim: null,
      rightDelim: null,
      size: "display",
      barSize: null
    };
  }
});
H({
  type: "infix",
  names: ["\\over", "\\choose", "\\atop", "\\brace", "\\brack"],
  props: {
    numArgs: 0,
    infix: !0
  },
  handler(n) {
    var {
      parser: e,
      funcName: t,
      token: r
    } = n, a;
    switch (t) {
      case "\\over":
        a = "\\frac";
        break;
      case "\\choose":
        a = "\\binom";
        break;
      case "\\atop":
        a = "\\\\atopfrac";
        break;
      case "\\brace":
        a = "\\\\bracefrac";
        break;
      case "\\brack":
        a = "\\\\brackfrac";
        break;
      default:
        throw new Error("Unrecognized infix genfrac command");
    }
    return {
      type: "infix",
      mode: e.mode,
      replaceWith: a,
      token: r
    };
  }
});
var Gi = ["display", "text", "script", "scriptscript"], Vi = function(e) {
  var t = null;
  return e.length > 0 && (t = e, t = t === "." ? null : t), t;
};
H({
  type: "genfrac",
  names: ["\\genfrac"],
  props: {
    numArgs: 6,
    allowedInArgument: !0,
    argTypes: ["math", "math", "size", "text", "math", "math"]
  },
  handler(n, e) {
    var {
      parser: t
    } = n, r = e[4], a = e[5], i = en(e[0]), l = i.type === "atom" && i.family === "open" ? Vi(i.text) : null, s = en(e[1]), u = s.type === "atom" && s.family === "close" ? Vi(s.text) : null, h = re(e[2], "size"), d, g = null;
    h.isBlank ? d = !0 : (g = h.value, d = g.number > 0);
    var p = "auto", v = e[3];
    if (v.type === "ordgroup") {
      if (v.body.length > 0) {
        var k = re(v.body[0], "textord");
        p = Gi[Number(k.text)];
      }
    } else
      v = re(v, "textord"), p = Gi[Number(v.text)];
    return {
      type: "genfrac",
      mode: t.mode,
      numer: r,
      denom: a,
      continued: !1,
      hasBarLine: d,
      barSize: g,
      leftDelim: l,
      rightDelim: u,
      size: p
    };
  },
  htmlBuilder: qa,
  mathmlBuilder: La
});
H({
  type: "infix",
  names: ["\\above"],
  props: {
    numArgs: 1,
    argTypes: ["size"],
    infix: !0
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r,
      token: a
    } = n;
    return {
      type: "infix",
      mode: t.mode,
      replaceWith: "\\\\abovefrac",
      size: re(e[0], "size").value,
      token: a
    };
  }
});
H({
  type: "genfrac",
  names: ["\\\\abovefrac"],
  props: {
    numArgs: 3,
    argTypes: ["math", "size", "math"]
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0], i = Ku(re(e[1], "infix").size), l = e[2], s = i.number > 0;
    return {
      type: "genfrac",
      mode: t.mode,
      numer: a,
      denom: l,
      continued: !1,
      hasBarLine: s,
      barSize: i,
      leftDelim: null,
      rightDelim: null,
      size: "auto"
    };
  },
  htmlBuilder: qa,
  mathmlBuilder: La
});
var oo = (n, e) => {
  var t = e.style, r, a;
  n.type === "supsub" ? (r = n.sup ? le(n.sup, e.havingStyle(t.sup()), e) : le(n.sub, e.havingStyle(t.sub()), e), a = re(n.base, "horizBrace")) : a = re(n, "horizBrace");
  var i = le(a.base, e.havingBaseStyle(Q.DISPLAY)), l = a0.svgSpan(a, e), s;
  if (a.isOver ? (s = F.makeVList({
    positionType: "firstBaseline",
    children: [{
      type: "elem",
      elem: i
    }, {
      type: "kern",
      size: 0.1
    }, {
      type: "elem",
      elem: l
    }]
  }, e), s.children[0].children[0].children[1].classes.push("svg-align")) : (s = F.makeVList({
    positionType: "bottom",
    positionData: i.depth + 0.1 + l.height,
    children: [{
      type: "elem",
      elem: l
    }, {
      type: "kern",
      size: 0.1
    }, {
      type: "elem",
      elem: i
    }]
  }, e), s.children[0].children[0].children[0].classes.push("svg-align")), r) {
    var u = F.makeSpan(["mord", a.isOver ? "mover" : "munder"], [s], e);
    a.isOver ? s = F.makeVList({
      positionType: "firstBaseline",
      children: [{
        type: "elem",
        elem: u
      }, {
        type: "kern",
        size: 0.2
      }, {
        type: "elem",
        elem: r
      }]
    }, e) : s = F.makeVList({
      positionType: "bottom",
      positionData: u.depth + 0.2 + r.height + r.depth,
      children: [{
        type: "elem",
        elem: r
      }, {
        type: "kern",
        size: 0.2
      }, {
        type: "elem",
        elem: u
      }]
    }, e);
  }
  return F.makeSpan(["mord", a.isOver ? "mover" : "munder"], [s], e);
}, wc = (n, e) => {
  var t = a0.mathMLnode(n.label);
  return new q.MathNode(n.isOver ? "mover" : "munder", [pe(n.base, e), t]);
};
H({
  type: "horizBrace",
  names: ["\\overbrace", "\\underbrace"],
  props: {
    numArgs: 1
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r
    } = n;
    return {
      type: "horizBrace",
      mode: t.mode,
      label: r,
      isOver: /^\\over/.test(r),
      base: e[0]
    };
  },
  htmlBuilder: oo,
  mathmlBuilder: wc
});
H({
  type: "href",
  names: ["\\href"],
  props: {
    numArgs: 2,
    argTypes: ["url", "original"],
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n, r = e[1], a = re(e[0], "url").url;
    return t.settings.isTrusted({
      command: "\\href",
      url: a
    }) ? {
      type: "href",
      mode: t.mode,
      href: a,
      body: Fe(r)
    } : t.formatUnsupportedCmd("\\href");
  },
  htmlBuilder: (n, e) => {
    var t = Me(n.body, e, !1);
    return F.makeAnchor(n.href, [], t, e);
  },
  mathmlBuilder: (n, e) => {
    var t = f0(n.body, e);
    return t instanceof tt || (t = new tt("mrow", [t])), t.setAttribute("href", n.href), t;
  }
});
H({
  type: "href",
  names: ["\\url"],
  props: {
    numArgs: 1,
    argTypes: ["url"],
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n, r = re(e[0], "url").url;
    if (!t.settings.isTrusted({
      command: "\\url",
      url: r
    }))
      return t.formatUnsupportedCmd("\\url");
    for (var a = [], i = 0; i < r.length; i++) {
      var l = r[i];
      l === "~" && (l = "\\textasciitilde"), a.push({
        type: "textord",
        mode: "text",
        text: l
      });
    }
    var s = {
      type: "text",
      mode: t.mode,
      font: "\\texttt",
      body: a
    };
    return {
      type: "href",
      mode: t.mode,
      href: r,
      body: Fe(s)
    };
  }
});
H({
  type: "hbox",
  names: ["\\hbox"],
  props: {
    numArgs: 1,
    argTypes: ["text"],
    allowedInText: !0,
    primitive: !0
  },
  handler(n, e) {
    var {
      parser: t
    } = n;
    return {
      type: "hbox",
      mode: t.mode,
      body: Fe(e[0])
    };
  },
  htmlBuilder(n, e) {
    var t = Me(n.body, e, !1);
    return F.makeFragment(t);
  },
  mathmlBuilder(n, e) {
    return new q.MathNode("mrow", Ze(n.body, e));
  }
});
H({
  type: "html",
  names: ["\\htmlClass", "\\htmlId", "\\htmlStyle", "\\htmlData"],
  props: {
    numArgs: 2,
    argTypes: ["raw", "original"],
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r,
      token: a
    } = n, i = re(e[0], "raw").string, l = e[1];
    t.settings.strict && t.settings.reportNonstrict("htmlExtension", "HTML extension is disabled on strict mode");
    var s, u = {};
    switch (r) {
      case "\\htmlClass":
        u.class = i, s = {
          command: "\\htmlClass",
          class: i
        };
        break;
      case "\\htmlId":
        u.id = i, s = {
          command: "\\htmlId",
          id: i
        };
        break;
      case "\\htmlStyle":
        u.style = i, s = {
          command: "\\htmlStyle",
          style: i
        };
        break;
      case "\\htmlData": {
        for (var h = i.split(","), d = 0; d < h.length; d++) {
          var g = h[d].split("=");
          if (g.length !== 2)
            throw new L("Error parsing key-value for \\htmlData");
          u["data-" + g[0].trim()] = g[1].trim();
        }
        s = {
          command: "\\htmlData",
          attributes: u
        };
        break;
      }
      default:
        throw new Error("Unrecognized html command");
    }
    return t.settings.isTrusted(s) ? {
      type: "html",
      mode: t.mode,
      attributes: u,
      body: Fe(l)
    } : t.formatUnsupportedCmd(r);
  },
  htmlBuilder: (n, e) => {
    var t = Me(n.body, e, !1), r = ["enclosing"];
    n.attributes.class && r.push(...n.attributes.class.trim().split(/\s+/));
    var a = F.makeSpan(r, t, e);
    for (var i in n.attributes)
      i !== "class" && n.attributes.hasOwnProperty(i) && a.setAttribute(i, n.attributes[i]);
    return a;
  },
  mathmlBuilder: (n, e) => f0(n.body, e)
});
H({
  type: "htmlmathml",
  names: ["\\html@mathml"],
  props: {
    numArgs: 2,
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n;
    return {
      type: "htmlmathml",
      mode: t.mode,
      html: Fe(e[0]),
      mathml: Fe(e[1])
    };
  },
  htmlBuilder: (n, e) => {
    var t = Me(n.html, e, !1);
    return F.makeFragment(t);
  },
  mathmlBuilder: (n, e) => f0(n.mathml, e)
});
var Rn = function(e) {
  if (/^[-+]? *(\d+(\.\d*)?|\.\d+)$/.test(e))
    return {
      number: +e,
      unit: "bp"
    };
  var t = /([-+]?) *(\d+(?:\.\d*)?|\.\d+) *([a-z]{2})/.exec(e);
  if (!t)
    throw new L("Invalid size: '" + e + "' in \\includegraphics");
  var r = {
    number: +(t[1] + t[2]),
    // sign + magnitude, cast to number
    unit: t[3]
  };
  if (!Es(r))
    throw new L("Invalid unit: '" + r.unit + "' in \\includegraphics.");
  return r;
};
H({
  type: "includegraphics",
  names: ["\\includegraphics"],
  props: {
    numArgs: 1,
    numOptionalArgs: 1,
    argTypes: ["raw", "url"],
    allowedInText: !1
  },
  handler: (n, e, t) => {
    var {
      parser: r
    } = n, a = {
      number: 0,
      unit: "em"
    }, i = {
      number: 0.9,
      unit: "em"
    }, l = {
      number: 0,
      unit: "em"
    }, s = "";
    if (t[0])
      for (var u = re(t[0], "raw").string, h = u.split(","), d = 0; d < h.length; d++) {
        var g = h[d].split("=");
        if (g.length === 2) {
          var p = g[1].trim();
          switch (g[0].trim()) {
            case "alt":
              s = p;
              break;
            case "width":
              a = Rn(p);
              break;
            case "height":
              i = Rn(p);
              break;
            case "totalheight":
              l = Rn(p);
              break;
            default:
              throw new L("Invalid key: '" + g[0] + "' in \\includegraphics.");
          }
        }
      }
    var v = re(e[0], "url").url;
    return s === "" && (s = v, s = s.replace(/^.*[\\/]/, ""), s = s.substring(0, s.lastIndexOf("."))), r.settings.isTrusted({
      command: "\\includegraphics",
      url: v
    }) ? {
      type: "includegraphics",
      mode: r.mode,
      alt: s,
      width: a,
      height: i,
      totalheight: l,
      src: v
    } : r.formatUnsupportedCmd("\\includegraphics");
  },
  htmlBuilder: (n, e) => {
    var t = ke(n.height, e), r = 0;
    n.totalheight.number > 0 && (r = ke(n.totalheight, e) - t);
    var a = 0;
    n.width.number > 0 && (a = ke(n.width, e));
    var i = {
      height: P(t + r)
    };
    a > 0 && (i.width = P(a)), r > 0 && (i.verticalAlign = P(-r));
    var l = new w1(n.src, n.alt, i);
    return l.height = t, l.depth = r, l;
  },
  mathmlBuilder: (n, e) => {
    var t = new q.MathNode("mglyph", []);
    t.setAttribute("alt", n.alt);
    var r = ke(n.height, e), a = 0;
    if (n.totalheight.number > 0 && (a = ke(n.totalheight, e) - r, t.setAttribute("valign", P(-a))), t.setAttribute("height", P(r + a)), n.width.number > 0) {
      var i = ke(n.width, e);
      t.setAttribute("width", P(i));
    }
    return t.setAttribute("src", n.src), t;
  }
});
H({
  type: "kern",
  names: ["\\kern", "\\mkern", "\\hskip", "\\mskip"],
  props: {
    numArgs: 1,
    argTypes: ["size"],
    primitive: !0,
    allowedInText: !0
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r
    } = n, a = re(e[0], "size");
    if (t.settings.strict) {
      var i = r[1] === "m", l = a.value.unit === "mu";
      i ? (l || t.settings.reportNonstrict("mathVsTextUnits", "LaTeX's " + r + " supports only mu units, " + ("not " + a.value.unit + " units")), t.mode !== "math" && t.settings.reportNonstrict("mathVsTextUnits", "LaTeX's " + r + " works only in math mode")) : l && t.settings.reportNonstrict("mathVsTextUnits", "LaTeX's " + r + " doesn't support mu units");
    }
    return {
      type: "kern",
      mode: t.mode,
      dimension: a.value
    };
  },
  htmlBuilder(n, e) {
    return F.makeGlue(n.dimension, e);
  },
  mathmlBuilder(n, e) {
    var t = ke(n.dimension, e);
    return new q.SpaceNode(t);
  }
});
H({
  type: "lap",
  names: ["\\mathllap", "\\mathrlap", "\\mathclap"],
  props: {
    numArgs: 1,
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0];
    return {
      type: "lap",
      mode: t.mode,
      alignment: r.slice(5),
      body: a
    };
  },
  htmlBuilder: (n, e) => {
    var t;
    n.alignment === "clap" ? (t = F.makeSpan([], [le(n.body, e)]), t = F.makeSpan(["inner"], [t], e)) : t = F.makeSpan(["inner"], [le(n.body, e)]);
    var r = F.makeSpan(["fix"], []), a = F.makeSpan([n.alignment], [t, r], e), i = F.makeSpan(["strut"]);
    return i.style.height = P(a.height + a.depth), a.depth && (i.style.verticalAlign = P(-a.depth)), a.children.unshift(i), a = F.makeSpan(["thinbox"], [a], e), F.makeSpan(["mord", "vbox"], [a], e);
  },
  mathmlBuilder: (n, e) => {
    var t = new q.MathNode("mpadded", [pe(n.body, e)]);
    if (n.alignment !== "rlap") {
      var r = n.alignment === "llap" ? "-1" : "-0.5";
      t.setAttribute("lspace", r + "width");
    }
    return t.setAttribute("width", "0px"), t;
  }
});
H({
  type: "styling",
  names: ["\\(", "$"],
  props: {
    numArgs: 0,
    allowedInText: !0,
    allowedInMath: !1
  },
  handler(n, e) {
    var {
      funcName: t,
      parser: r
    } = n, a = r.mode;
    r.switchMode("math");
    var i = t === "\\(" ? "\\)" : "$", l = r.parseExpression(!1, i);
    return r.expect(i), r.switchMode(a), {
      type: "styling",
      mode: r.mode,
      style: "text",
      body: l
    };
  }
});
H({
  type: "text",
  // Doesn't matter what this is.
  names: ["\\)", "\\]"],
  props: {
    numArgs: 0,
    allowedInText: !0,
    allowedInMath: !1
  },
  handler(n, e) {
    throw new L("Mismatched " + n.funcName);
  }
});
var Wi = (n, e) => {
  switch (e.style.size) {
    case Q.DISPLAY.size:
      return n.display;
    case Q.TEXT.size:
      return n.text;
    case Q.SCRIPT.size:
      return n.script;
    case Q.SCRIPTSCRIPT.size:
      return n.scriptscript;
    default:
      return n.text;
  }
};
H({
  type: "mathchoice",
  names: ["\\mathchoice"],
  props: {
    numArgs: 4,
    primitive: !0
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n;
    return {
      type: "mathchoice",
      mode: t.mode,
      display: Fe(e[0]),
      text: Fe(e[1]),
      script: Fe(e[2]),
      scriptscript: Fe(e[3])
    };
  },
  htmlBuilder: (n, e) => {
    var t = Wi(n, e), r = Me(t, e, !1);
    return F.makeFragment(r);
  },
  mathmlBuilder: (n, e) => {
    var t = Wi(n, e);
    return f0(t, e);
  }
});
var uo = (n, e, t, r, a, i, l) => {
  n = F.makeSpan([], [n]);
  var s = t && Z.isCharacterBox(t), u, h;
  if (e) {
    var d = le(e, r.havingStyle(a.sup()), r);
    h = {
      elem: d,
      kern: Math.max(r.fontMetrics().bigOpSpacing1, r.fontMetrics().bigOpSpacing3 - d.depth)
    };
  }
  if (t) {
    var g = le(t, r.havingStyle(a.sub()), r);
    u = {
      elem: g,
      kern: Math.max(r.fontMetrics().bigOpSpacing2, r.fontMetrics().bigOpSpacing4 - g.height)
    };
  }
  var p;
  if (h && u) {
    var v = r.fontMetrics().bigOpSpacing5 + u.elem.height + u.elem.depth + u.kern + n.depth + l;
    p = F.makeVList({
      positionType: "bottom",
      positionData: v,
      children: [{
        type: "kern",
        size: r.fontMetrics().bigOpSpacing5
      }, {
        type: "elem",
        elem: u.elem,
        marginLeft: P(-i)
      }, {
        type: "kern",
        size: u.kern
      }, {
        type: "elem",
        elem: n
      }, {
        type: "kern",
        size: h.kern
      }, {
        type: "elem",
        elem: h.elem,
        marginLeft: P(i)
      }, {
        type: "kern",
        size: r.fontMetrics().bigOpSpacing5
      }]
    }, r);
  } else if (u) {
    var k = n.height - l;
    p = F.makeVList({
      positionType: "top",
      positionData: k,
      children: [{
        type: "kern",
        size: r.fontMetrics().bigOpSpacing5
      }, {
        type: "elem",
        elem: u.elem,
        marginLeft: P(-i)
      }, {
        type: "kern",
        size: u.kern
      }, {
        type: "elem",
        elem: n
      }]
    }, r);
  } else if (h) {
    var A = n.depth + l;
    p = F.makeVList({
      positionType: "bottom",
      positionData: A,
      children: [{
        type: "elem",
        elem: n
      }, {
        type: "kern",
        size: h.kern
      }, {
        type: "elem",
        elem: h.elem,
        marginLeft: P(i)
      }, {
        type: "kern",
        size: r.fontMetrics().bigOpSpacing5
      }]
    }, r);
  } else
    return n;
  var C = [p];
  if (u && i !== 0 && !s) {
    var z = F.makeSpan(["mspace"], [], r);
    z.style.marginRight = P(i), C.unshift(z);
  }
  return F.makeSpan(["mop", "op-limits"], C, r);
}, co = ["\\smallint"], O0 = (n, e) => {
  var t, r, a = !1, i;
  n.type === "supsub" ? (t = n.sup, r = n.sub, i = re(n.base, "op"), a = !0) : i = re(n, "op");
  var l = e.style, s = !1;
  l.size === Q.DISPLAY.size && i.symbol && !Z.contains(co, i.name) && (s = !0);
  var u;
  if (i.symbol) {
    var h = s ? "Size2-Regular" : "Size1-Regular", d = "";
    if ((i.name === "\\oiint" || i.name === "\\oiiint") && (d = i.name.slice(1), i.name = d === "oiint" ? "\\iint" : "\\iiint"), u = F.makeSymbol(i.name, h, "math", e, ["mop", "op-symbol", s ? "large-op" : "small-op"]), d.length > 0) {
      var g = u.italic, p = F.staticSvg(d + "Size" + (s ? "2" : "1"), e);
      u = F.makeVList({
        positionType: "individualShift",
        children: [{
          type: "elem",
          elem: u,
          shift: 0
        }, {
          type: "elem",
          elem: p,
          shift: s ? 0.08 : 0
        }]
      }, e), i.name = "\\" + d, u.classes.unshift("mop"), u.italic = g;
    }
  } else if (i.body) {
    var v = Me(i.body, e, !0);
    v.length === 1 && v[0] instanceof ut ? (u = v[0], u.classes[0] = "mop") : u = F.makeSpan(["mop"], v, e);
  } else {
    for (var k = [], A = 1; A < i.name.length; A++)
      k.push(F.mathsym(i.name[A], i.mode, e));
    u = F.makeSpan(["mop"], k, e);
  }
  var C = 0, z = 0;
  return (u instanceof ut || i.name === "\\oiint" || i.name === "\\oiiint") && !i.suppressBaseShift && (C = (u.height - u.depth) / 2 - e.fontMetrics().axisHeight, z = u.italic), a ? uo(u, t, r, e, l, z, C) : (C && (u.style.position = "relative", u.style.top = P(C)), u);
}, hr = (n, e) => {
  var t;
  if (n.symbol)
    t = new tt("mo", [ct(n.name, n.mode)]), Z.contains(co, n.name) && t.setAttribute("largeop", "false");
  else if (n.body)
    t = new tt("mo", Ze(n.body, e));
  else {
    t = new tt("mi", [new Bt(n.name.slice(1))]);
    var r = new tt("mo", [ct("⁡", "text")]);
    n.parentIsSupSub ? t = new tt("mrow", [t, r]) : t = Ls([t, r]);
  }
  return t;
}, xc = {
  "∏": "\\prod",
  "∐": "\\coprod",
  "∑": "\\sum",
  "⋀": "\\bigwedge",
  "⋁": "\\bigvee",
  "⋂": "\\bigcap",
  "⋃": "\\bigcup",
  "⨀": "\\bigodot",
  "⨁": "\\bigoplus",
  "⨂": "\\bigotimes",
  "⨄": "\\biguplus",
  "⨆": "\\bigsqcup"
};
H({
  type: "op",
  names: ["\\coprod", "\\bigvee", "\\bigwedge", "\\biguplus", "\\bigcap", "\\bigcup", "\\intop", "\\prod", "\\sum", "\\bigotimes", "\\bigoplus", "\\bigodot", "\\bigsqcup", "\\smallint", "∏", "∐", "∑", "⋀", "⋁", "⋂", "⋃", "⨀", "⨁", "⨂", "⨄", "⨆"],
  props: {
    numArgs: 0
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r
    } = n, a = r;
    return a.length === 1 && (a = xc[a]), {
      type: "op",
      mode: t.mode,
      limits: !0,
      parentIsSupSub: !1,
      symbol: !0,
      name: a
    };
  },
  htmlBuilder: O0,
  mathmlBuilder: hr
});
H({
  type: "op",
  names: ["\\mathop"],
  props: {
    numArgs: 1,
    primitive: !0
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n, r = e[0];
    return {
      type: "op",
      mode: t.mode,
      limits: !1,
      parentIsSupSub: !1,
      symbol: !1,
      body: Fe(r)
    };
  },
  htmlBuilder: O0,
  mathmlBuilder: hr
});
var kc = {
  "∫": "\\int",
  "∬": "\\iint",
  "∭": "\\iiint",
  "∮": "\\oint",
  "∯": "\\oiint",
  "∰": "\\oiiint"
};
H({
  type: "op",
  names: ["\\arcsin", "\\arccos", "\\arctan", "\\arctg", "\\arcctg", "\\arg", "\\ch", "\\cos", "\\cosec", "\\cosh", "\\cot", "\\cotg", "\\coth", "\\csc", "\\ctg", "\\cth", "\\deg", "\\dim", "\\exp", "\\hom", "\\ker", "\\lg", "\\ln", "\\log", "\\sec", "\\sin", "\\sinh", "\\sh", "\\tan", "\\tanh", "\\tg", "\\th"],
  props: {
    numArgs: 0
  },
  handler(n) {
    var {
      parser: e,
      funcName: t
    } = n;
    return {
      type: "op",
      mode: e.mode,
      limits: !1,
      parentIsSupSub: !1,
      symbol: !1,
      name: t
    };
  },
  htmlBuilder: O0,
  mathmlBuilder: hr
});
H({
  type: "op",
  names: ["\\det", "\\gcd", "\\inf", "\\lim", "\\max", "\\min", "\\Pr", "\\sup"],
  props: {
    numArgs: 0
  },
  handler(n) {
    var {
      parser: e,
      funcName: t
    } = n;
    return {
      type: "op",
      mode: e.mode,
      limits: !0,
      parentIsSupSub: !1,
      symbol: !1,
      name: t
    };
  },
  htmlBuilder: O0,
  mathmlBuilder: hr
});
H({
  type: "op",
  names: ["\\int", "\\iint", "\\iiint", "\\oint", "\\oiint", "\\oiiint", "∫", "∬", "∭", "∮", "∯", "∰"],
  props: {
    numArgs: 0
  },
  handler(n) {
    var {
      parser: e,
      funcName: t
    } = n, r = t;
    return r.length === 1 && (r = kc[r]), {
      type: "op",
      mode: e.mode,
      limits: !1,
      parentIsSupSub: !1,
      symbol: !0,
      name: r
    };
  },
  htmlBuilder: O0,
  mathmlBuilder: hr
});
var ho = (n, e) => {
  var t, r, a = !1, i;
  n.type === "supsub" ? (t = n.sup, r = n.sub, i = re(n.base, "operatorname"), a = !0) : i = re(n, "operatorname");
  var l;
  if (i.body.length > 0) {
    for (var s = i.body.map((g) => {
      var p = g.text;
      return typeof p == "string" ? {
        type: "textord",
        mode: g.mode,
        text: p
      } : g;
    }), u = Me(s, e.withFont("mathrm"), !0), h = 0; h < u.length; h++) {
      var d = u[h];
      d instanceof ut && (d.text = d.text.replace(/\u2212/, "-").replace(/\u2217/, "*"));
    }
    l = F.makeSpan(["mop"], u, e);
  } else
    l = F.makeSpan(["mop"], [], e);
  return a ? uo(l, t, r, e, e.style, 0, 0) : l;
}, Dc = (n, e) => {
  for (var t = Ze(n.body, e.withFont("mathrm")), r = !0, a = 0; a < t.length; a++) {
    var i = t[a];
    if (!(i instanceof q.SpaceNode)) if (i instanceof q.MathNode)
      switch (i.type) {
        case "mi":
        case "mn":
        case "ms":
        case "mspace":
        case "mtext":
          break;
        case "mo": {
          var l = i.children[0];
          i.children.length === 1 && l instanceof q.TextNode ? l.text = l.text.replace(/\u2212/, "-").replace(/\u2217/, "*") : r = !1;
          break;
        }
        default:
          r = !1;
      }
    else
      r = !1;
  }
  if (r) {
    var s = t.map((d) => d.toText()).join("");
    t = [new q.TextNode(s)];
  }
  var u = new q.MathNode("mi", t);
  u.setAttribute("mathvariant", "normal");
  var h = new q.MathNode("mo", [ct("⁡", "text")]);
  return n.parentIsSupSub ? new q.MathNode("mrow", [u, h]) : q.newDocumentFragment([u, h]);
};
H({
  type: "operatorname",
  names: ["\\operatorname@", "\\operatornamewithlimits"],
  props: {
    numArgs: 1
  },
  handler: (n, e) => {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0];
    return {
      type: "operatorname",
      mode: t.mode,
      body: Fe(a),
      alwaysHandleSupSub: r === "\\operatornamewithlimits",
      limits: !1,
      parentIsSupSub: !1
    };
  },
  htmlBuilder: ho,
  mathmlBuilder: Dc
});
f("\\operatorname", "\\@ifstar\\operatornamewithlimits\\operatorname@");
x0({
  type: "ordgroup",
  htmlBuilder(n, e) {
    return n.semisimple ? F.makeFragment(Me(n.body, e, !1)) : F.makeSpan(["mord"], Me(n.body, e, !0), e);
  },
  mathmlBuilder(n, e) {
    return f0(n.body, e, !0);
  }
});
H({
  type: "overline",
  names: ["\\overline"],
  props: {
    numArgs: 1
  },
  handler(n, e) {
    var {
      parser: t
    } = n, r = e[0];
    return {
      type: "overline",
      mode: t.mode,
      body: r
    };
  },
  htmlBuilder(n, e) {
    var t = le(n.body, e.havingCrampedStyle()), r = F.makeLineSpan("overline-line", e), a = e.fontMetrics().defaultRuleThickness, i = F.makeVList({
      positionType: "firstBaseline",
      children: [{
        type: "elem",
        elem: t
      }, {
        type: "kern",
        size: 3 * a
      }, {
        type: "elem",
        elem: r
      }, {
        type: "kern",
        size: a
      }]
    }, e);
    return F.makeSpan(["mord", "overline"], [i], e);
  },
  mathmlBuilder(n, e) {
    var t = new q.MathNode("mo", [new q.TextNode("‾")]);
    t.setAttribute("stretchy", "true");
    var r = new q.MathNode("mover", [pe(n.body, e), t]);
    return r.setAttribute("accent", "true"), r;
  }
});
H({
  type: "phantom",
  names: ["\\phantom"],
  props: {
    numArgs: 1,
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n, r = e[0];
    return {
      type: "phantom",
      mode: t.mode,
      body: Fe(r)
    };
  },
  htmlBuilder: (n, e) => {
    var t = Me(n.body, e.withPhantom(), !1);
    return F.makeFragment(t);
  },
  mathmlBuilder: (n, e) => {
    var t = Ze(n.body, e);
    return new q.MathNode("mphantom", t);
  }
});
H({
  type: "hphantom",
  names: ["\\hphantom"],
  props: {
    numArgs: 1,
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n, r = e[0];
    return {
      type: "hphantom",
      mode: t.mode,
      body: r
    };
  },
  htmlBuilder: (n, e) => {
    var t = F.makeSpan([], [le(n.body, e.withPhantom())]);
    if (t.height = 0, t.depth = 0, t.children)
      for (var r = 0; r < t.children.length; r++)
        t.children[r].height = 0, t.children[r].depth = 0;
    return t = F.makeVList({
      positionType: "firstBaseline",
      children: [{
        type: "elem",
        elem: t
      }]
    }, e), F.makeSpan(["mord"], [t], e);
  },
  mathmlBuilder: (n, e) => {
    var t = Ze(Fe(n.body), e), r = new q.MathNode("mphantom", t), a = new q.MathNode("mpadded", [r]);
    return a.setAttribute("height", "0px"), a.setAttribute("depth", "0px"), a;
  }
});
H({
  type: "vphantom",
  names: ["\\vphantom"],
  props: {
    numArgs: 1,
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      parser: t
    } = n, r = e[0];
    return {
      type: "vphantom",
      mode: t.mode,
      body: r
    };
  },
  htmlBuilder: (n, e) => {
    var t = F.makeSpan(["inner"], [le(n.body, e.withPhantom())]), r = F.makeSpan(["fix"], []);
    return F.makeSpan(["mord", "rlap"], [t, r], e);
  },
  mathmlBuilder: (n, e) => {
    var t = Ze(Fe(n.body), e), r = new q.MathNode("mphantom", t), a = new q.MathNode("mpadded", [r]);
    return a.setAttribute("width", "0px"), a;
  }
});
H({
  type: "raisebox",
  names: ["\\raisebox"],
  props: {
    numArgs: 2,
    argTypes: ["size", "hbox"],
    allowedInText: !0
  },
  handler(n, e) {
    var {
      parser: t
    } = n, r = re(e[0], "size").value, a = e[1];
    return {
      type: "raisebox",
      mode: t.mode,
      dy: r,
      body: a
    };
  },
  htmlBuilder(n, e) {
    var t = le(n.body, e), r = ke(n.dy, e);
    return F.makeVList({
      positionType: "shift",
      positionData: -r,
      children: [{
        type: "elem",
        elem: t
      }]
    }, e);
  },
  mathmlBuilder(n, e) {
    var t = new q.MathNode("mpadded", [pe(n.body, e)]), r = n.dy.number + n.dy.unit;
    return t.setAttribute("voffset", r), t;
  }
});
H({
  type: "internal",
  names: ["\\relax"],
  props: {
    numArgs: 0,
    allowedInText: !0,
    allowedInArgument: !0
  },
  handler(n) {
    var {
      parser: e
    } = n;
    return {
      type: "internal",
      mode: e.mode
    };
  }
});
H({
  type: "rule",
  names: ["\\rule"],
  props: {
    numArgs: 2,
    numOptionalArgs: 1,
    allowedInText: !0,
    allowedInMath: !0,
    argTypes: ["size", "size", "size"]
  },
  handler(n, e, t) {
    var {
      parser: r
    } = n, a = t[0], i = re(e[0], "size"), l = re(e[1], "size");
    return {
      type: "rule",
      mode: r.mode,
      shift: a && re(a, "size").value,
      width: i.value,
      height: l.value
    };
  },
  htmlBuilder(n, e) {
    var t = F.makeSpan(["mord", "rule"], [], e), r = ke(n.width, e), a = ke(n.height, e), i = n.shift ? ke(n.shift, e) : 0;
    return t.style.borderRightWidth = P(r), t.style.borderTopWidth = P(a), t.style.bottom = P(i), t.width = r, t.height = a + i, t.depth = -i, t.maxFontSize = a * 1.125 * e.sizeMultiplier, t;
  },
  mathmlBuilder(n, e) {
    var t = ke(n.width, e), r = ke(n.height, e), a = n.shift ? ke(n.shift, e) : 0, i = e.color && e.getColor() || "black", l = new q.MathNode("mspace");
    l.setAttribute("mathbackground", i), l.setAttribute("width", P(t)), l.setAttribute("height", P(r));
    var s = new q.MathNode("mpadded", [l]);
    return a >= 0 ? s.setAttribute("height", P(a)) : (s.setAttribute("height", P(a)), s.setAttribute("depth", P(-a))), s.setAttribute("voffset", P(a)), s;
  }
});
function mo(n, e, t) {
  for (var r = Me(n, e, !1), a = e.sizeMultiplier / t.sizeMultiplier, i = 0; i < r.length; i++) {
    var l = r[i].classes.indexOf("sizing");
    l < 0 ? Array.prototype.push.apply(r[i].classes, e.sizingClasses(t)) : r[i].classes[l + 1] === "reset-size" + e.size && (r[i].classes[l + 1] = "reset-size" + t.size), r[i].height *= a, r[i].depth *= a;
  }
  return F.makeFragment(r);
}
var ji = ["\\tiny", "\\sixptsize", "\\scriptsize", "\\footnotesize", "\\small", "\\normalsize", "\\large", "\\Large", "\\LARGE", "\\huge", "\\Huge"], Sc = (n, e) => {
  var t = e.havingSize(n.size);
  return mo(n.body, t, e);
};
H({
  type: "sizing",
  names: ji,
  props: {
    numArgs: 0,
    allowedInText: !0
  },
  handler: (n, e) => {
    var {
      breakOnTokenText: t,
      funcName: r,
      parser: a
    } = n, i = a.parseExpression(!1, t);
    return {
      type: "sizing",
      mode: a.mode,
      // Figure out what size to use based on the list of functions above
      size: ji.indexOf(r) + 1,
      body: i
    };
  },
  htmlBuilder: Sc,
  mathmlBuilder: (n, e) => {
    var t = e.havingSize(n.size), r = Ze(n.body, t), a = new q.MathNode("mstyle", r);
    return a.setAttribute("mathsize", P(t.sizeMultiplier)), a;
  }
});
H({
  type: "smash",
  names: ["\\smash"],
  props: {
    numArgs: 1,
    numOptionalArgs: 1,
    allowedInText: !0
  },
  handler: (n, e, t) => {
    var {
      parser: r
    } = n, a = !1, i = !1, l = t[0] && re(t[0], "ordgroup");
    if (l)
      for (var s = "", u = 0; u < l.body.length; ++u) {
        var h = l.body[u];
        if (s = h.text, s === "t")
          a = !0;
        else if (s === "b")
          i = !0;
        else {
          a = !1, i = !1;
          break;
        }
      }
    else
      a = !0, i = !0;
    var d = e[0];
    return {
      type: "smash",
      mode: r.mode,
      body: d,
      smashHeight: a,
      smashDepth: i
    };
  },
  htmlBuilder: (n, e) => {
    var t = F.makeSpan([], [le(n.body, e)]);
    if (!n.smashHeight && !n.smashDepth)
      return t;
    if (n.smashHeight && (t.height = 0, t.children))
      for (var r = 0; r < t.children.length; r++)
        t.children[r].height = 0;
    if (n.smashDepth && (t.depth = 0, t.children))
      for (var a = 0; a < t.children.length; a++)
        t.children[a].depth = 0;
    var i = F.makeVList({
      positionType: "firstBaseline",
      children: [{
        type: "elem",
        elem: t
      }]
    }, e);
    return F.makeSpan(["mord"], [i], e);
  },
  mathmlBuilder: (n, e) => {
    var t = new q.MathNode("mpadded", [pe(n.body, e)]);
    return n.smashHeight && t.setAttribute("height", "0px"), n.smashDepth && t.setAttribute("depth", "0px"), t;
  }
});
H({
  type: "sqrt",
  names: ["\\sqrt"],
  props: {
    numArgs: 1,
    numOptionalArgs: 1
  },
  handler(n, e, t) {
    var {
      parser: r
    } = n, a = t[0], i = e[0];
    return {
      type: "sqrt",
      mode: r.mode,
      body: i,
      index: a
    };
  },
  htmlBuilder(n, e) {
    var t = le(n.body, e.havingCrampedStyle());
    t.height === 0 && (t.height = e.fontMetrics().xHeight), t = F.wrapFragment(t, e);
    var r = e.fontMetrics(), a = r.defaultRuleThickness, i = a;
    e.style.id < Q.TEXT.id && (i = e.fontMetrics().xHeight);
    var l = a + i / 4, s = t.height + t.depth + l + a, {
      span: u,
      ruleWidth: h,
      advanceWidth: d
    } = t0.sqrtImage(s, e), g = u.height - h;
    g > t.height + t.depth + l && (l = (l + g - t.height - t.depth) / 2);
    var p = u.height - t.height - l - h;
    t.style.paddingLeft = P(d);
    var v = F.makeVList({
      positionType: "firstBaseline",
      children: [{
        type: "elem",
        elem: t,
        wrapperClasses: ["svg-align"]
      }, {
        type: "kern",
        size: -(t.height + p)
      }, {
        type: "elem",
        elem: u
      }, {
        type: "kern",
        size: h
      }]
    }, e);
    if (n.index) {
      var k = e.havingStyle(Q.SCRIPTSCRIPT), A = le(n.index, k, e), C = 0.6 * (v.height - v.depth), z = F.makeVList({
        positionType: "shift",
        positionData: -C,
        children: [{
          type: "elem",
          elem: A
        }]
      }, e), x = F.makeSpan(["root"], [z]);
      return F.makeSpan(["mord", "sqrt"], [x, v], e);
    } else
      return F.makeSpan(["mord", "sqrt"], [v], e);
  },
  mathmlBuilder(n, e) {
    var {
      body: t,
      index: r
    } = n;
    return r ? new q.MathNode("mroot", [pe(t, e), pe(r, e)]) : new q.MathNode("msqrt", [pe(t, e)]);
  }
});
var Yi = {
  display: Q.DISPLAY,
  text: Q.TEXT,
  script: Q.SCRIPT,
  scriptscript: Q.SCRIPTSCRIPT
};
H({
  type: "styling",
  names: ["\\displaystyle", "\\textstyle", "\\scriptstyle", "\\scriptscriptstyle"],
  props: {
    numArgs: 0,
    allowedInText: !0,
    primitive: !0
  },
  handler(n, e) {
    var {
      breakOnTokenText: t,
      funcName: r,
      parser: a
    } = n, i = a.parseExpression(!0, t), l = r.slice(1, r.length - 5);
    return {
      type: "styling",
      mode: a.mode,
      // Figure out what style to use by pulling out the style from
      // the function name
      style: l,
      body: i
    };
  },
  htmlBuilder(n, e) {
    var t = Yi[n.style], r = e.havingStyle(t).withFont("");
    return mo(n.body, r, e);
  },
  mathmlBuilder(n, e) {
    var t = Yi[n.style], r = e.havingStyle(t), a = Ze(n.body, r), i = new q.MathNode("mstyle", a), l = {
      display: ["0", "true"],
      text: ["0", "false"],
      script: ["1", "false"],
      scriptscript: ["2", "false"]
    }, s = l[n.style];
    return i.setAttribute("scriptlevel", s[0]), i.setAttribute("displaystyle", s[1]), i;
  }
});
var Ac = function(e, t) {
  var r = e.base;
  if (r)
    if (r.type === "op") {
      var a = r.limits && (t.style.size === Q.DISPLAY.size || r.alwaysHandleSupSub);
      return a ? O0 : null;
    } else if (r.type === "operatorname") {
      var i = r.alwaysHandleSupSub && (t.style.size === Q.DISPLAY.size || r.limits);
      return i ? ho : null;
    } else {
      if (r.type === "accent")
        return Z.isCharacterBox(r.base) ? $a : null;
      if (r.type === "horizBrace") {
        var l = !e.sub;
        return l === r.isOver ? oo : null;
      } else
        return null;
    }
  else return null;
};
x0({
  type: "supsub",
  htmlBuilder(n, e) {
    var t = Ac(n, e);
    if (t)
      return t(n, e);
    var {
      base: r,
      sup: a,
      sub: i
    } = n, l = le(r, e), s, u, h = e.fontMetrics(), d = 0, g = 0, p = r && Z.isCharacterBox(r);
    if (a) {
      var v = e.havingStyle(e.style.sup());
      s = le(a, v, e), p || (d = l.height - v.fontMetrics().supDrop * v.sizeMultiplier / e.sizeMultiplier);
    }
    if (i) {
      var k = e.havingStyle(e.style.sub());
      u = le(i, k, e), p || (g = l.depth + k.fontMetrics().subDrop * k.sizeMultiplier / e.sizeMultiplier);
    }
    var A;
    e.style === Q.DISPLAY ? A = h.sup1 : e.style.cramped ? A = h.sup3 : A = h.sup2;
    var C = e.sizeMultiplier, z = P(0.5 / h.ptPerEm / C), x = null;
    if (u) {
      var _ = n.base && n.base.type === "op" && n.base.name && (n.base.name === "\\oiint" || n.base.name === "\\oiiint");
      (l instanceof ut || _) && (x = P(-l.italic));
    }
    var w;
    if (s && u) {
      d = Math.max(d, A, s.depth + 0.25 * h.xHeight), g = Math.max(g, h.sub2);
      var E = h.defaultRuleThickness, T = 4 * E;
      if (d - s.depth - (u.height - g) < T) {
        g = T - (d - s.depth) + u.height;
        var $ = 0.8 * h.xHeight - (d - s.depth);
        $ > 0 && (d += $, g -= $);
      }
      var M = [{
        type: "elem",
        elem: u,
        shift: g,
        marginRight: z,
        marginLeft: x
      }, {
        type: "elem",
        elem: s,
        shift: -d,
        marginRight: z
      }];
      w = F.makeVList({
        positionType: "individualShift",
        children: M
      }, e);
    } else if (u) {
      g = Math.max(g, h.sub1, u.height - 0.8 * h.xHeight);
      var B = [{
        type: "elem",
        elem: u,
        marginLeft: x,
        marginRight: z
      }];
      w = F.makeVList({
        positionType: "shift",
        positionData: g,
        children: B
      }, e);
    } else if (s)
      d = Math.max(d, A, s.depth + 0.25 * h.xHeight), w = F.makeVList({
        positionType: "shift",
        positionData: -d,
        children: [{
          type: "elem",
          elem: s,
          marginRight: z
        }]
      }, e);
    else
      throw new Error("supsub must have either sup or sub.");
    var G = aa(l, "right") || "mord";
    return F.makeSpan([G], [l, F.makeSpan(["msupsub"], [w])], e);
  },
  mathmlBuilder(n, e) {
    var t = !1, r, a;
    n.base && n.base.type === "horizBrace" && (a = !!n.sup, a === n.base.isOver && (t = !0, r = n.base.isOver)), n.base && (n.base.type === "op" || n.base.type === "operatorname") && (n.base.parentIsSupSub = !0);
    var i = [pe(n.base, e)];
    n.sub && i.push(pe(n.sub, e)), n.sup && i.push(pe(n.sup, e));
    var l;
    if (t)
      l = r ? "mover" : "munder";
    else if (n.sub)
      if (n.sup) {
        var h = n.base;
        h && h.type === "op" && h.limits && e.style === Q.DISPLAY || h && h.type === "operatorname" && h.alwaysHandleSupSub && (e.style === Q.DISPLAY || h.limits) ? l = "munderover" : l = "msubsup";
      } else {
        var u = n.base;
        u && u.type === "op" && u.limits && (e.style === Q.DISPLAY || u.alwaysHandleSupSub) || u && u.type === "operatorname" && u.alwaysHandleSupSub && (u.limits || e.style === Q.DISPLAY) ? l = "munder" : l = "msub";
      }
    else {
      var s = n.base;
      s && s.type === "op" && s.limits && (e.style === Q.DISPLAY || s.alwaysHandleSupSub) || s && s.type === "operatorname" && s.alwaysHandleSupSub && (s.limits || e.style === Q.DISPLAY) ? l = "mover" : l = "msup";
    }
    return new q.MathNode(l, i);
  }
});
x0({
  type: "atom",
  htmlBuilder(n, e) {
    return F.mathsym(n.text, n.mode, e, ["m" + n.family]);
  },
  mathmlBuilder(n, e) {
    var t = new q.MathNode("mo", [ct(n.text, n.mode)]);
    if (n.family === "bin") {
      var r = Ca(n, e);
      r === "bold-italic" && t.setAttribute("mathvariant", r);
    } else n.family === "punct" ? t.setAttribute("separator", "true") : (n.family === "open" || n.family === "close") && t.setAttribute("stretchy", "false");
    return t;
  }
});
var fo = {
  mi: "italic",
  mn: "normal",
  mtext: "normal"
};
x0({
  type: "mathord",
  htmlBuilder(n, e) {
    return F.makeOrd(n, e, "mathord");
  },
  mathmlBuilder(n, e) {
    var t = new q.MathNode("mi", [ct(n.text, n.mode, e)]), r = Ca(n, e) || "italic";
    return r !== fo[t.type] && t.setAttribute("mathvariant", r), t;
  }
});
x0({
  type: "textord",
  htmlBuilder(n, e) {
    return F.makeOrd(n, e, "textord");
  },
  mathmlBuilder(n, e) {
    var t = ct(n.text, n.mode, e), r = Ca(n, e) || "normal", a;
    return n.mode === "text" ? a = new q.MathNode("mtext", [t]) : /[0-9]/.test(n.text) ? a = new q.MathNode("mn", [t]) : n.text === "\\prime" ? a = new q.MathNode("mo", [t]) : a = new q.MathNode("mi", [t]), r !== fo[a.type] && a.setAttribute("mathvariant", r), a;
  }
});
var Nn = {
  "\\nobreak": "nobreak",
  "\\allowbreak": "allowbreak"
}, qn = {
  " ": {},
  "\\ ": {},
  "~": {
    className: "nobreak"
  },
  "\\space": {},
  "\\nobreakspace": {
    className: "nobreak"
  }
};
x0({
  type: "spacing",
  htmlBuilder(n, e) {
    if (qn.hasOwnProperty(n.text)) {
      var t = qn[n.text].className || "";
      if (n.mode === "text") {
        var r = F.makeOrd(n, e, "textord");
        return r.classes.push(t), r;
      } else
        return F.makeSpan(["mspace", t], [F.mathsym(n.text, n.mode, e)], e);
    } else {
      if (Nn.hasOwnProperty(n.text))
        return F.makeSpan(["mspace", Nn[n.text]], [], e);
      throw new L('Unknown type of space "' + n.text + '"');
    }
  },
  mathmlBuilder(n, e) {
    var t;
    if (qn.hasOwnProperty(n.text))
      t = new q.MathNode("mtext", [new q.TextNode(" ")]);
    else {
      if (Nn.hasOwnProperty(n.text))
        return new q.MathNode("mspace");
      throw new L('Unknown type of space "' + n.text + '"');
    }
    return t;
  }
});
var Xi = () => {
  var n = new q.MathNode("mtd", []);
  return n.setAttribute("width", "50%"), n;
};
x0({
  type: "tag",
  mathmlBuilder(n, e) {
    var t = new q.MathNode("mtable", [new q.MathNode("mtr", [Xi(), new q.MathNode("mtd", [f0(n.body, e)]), Xi(), new q.MathNode("mtd", [f0(n.tag, e)])])]);
    return t.setAttribute("width", "100%"), t;
  }
});
var Zi = {
  "\\text": void 0,
  "\\textrm": "textrm",
  "\\textsf": "textsf",
  "\\texttt": "texttt",
  "\\textnormal": "textrm"
}, Ki = {
  "\\textbf": "textbf",
  "\\textmd": "textmd"
}, Ec = {
  "\\textit": "textit",
  "\\textup": "textup"
}, Qi = (n, e) => {
  var t = n.font;
  if (t) {
    if (Zi[t])
      return e.withTextFontFamily(Zi[t]);
    if (Ki[t])
      return e.withTextFontWeight(Ki[t]);
    if (t === "\\emph")
      return e.fontShape === "textit" ? e.withTextFontShape("textup") : e.withTextFontShape("textit");
  } else return e;
  return e.withTextFontShape(Ec[t]);
};
H({
  type: "text",
  names: [
    // Font families
    "\\text",
    "\\textrm",
    "\\textsf",
    "\\texttt",
    "\\textnormal",
    // Font weights
    "\\textbf",
    "\\textmd",
    // Font Shapes
    "\\textit",
    "\\textup",
    "\\emph"
  ],
  props: {
    numArgs: 1,
    argTypes: ["text"],
    allowedInArgument: !0,
    allowedInText: !0
  },
  handler(n, e) {
    var {
      parser: t,
      funcName: r
    } = n, a = e[0];
    return {
      type: "text",
      mode: t.mode,
      body: Fe(a),
      font: r
    };
  },
  htmlBuilder(n, e) {
    var t = Qi(n, e), r = Me(n.body, t, !0);
    return F.makeSpan(["mord", "text"], r, t);
  },
  mathmlBuilder(n, e) {
    var t = Qi(n, e);
    return f0(n.body, t);
  }
});
H({
  type: "underline",
  names: ["\\underline"],
  props: {
    numArgs: 1,
    allowedInText: !0
  },
  handler(n, e) {
    var {
      parser: t
    } = n;
    return {
      type: "underline",
      mode: t.mode,
      body: e[0]
    };
  },
  htmlBuilder(n, e) {
    var t = le(n.body, e), r = F.makeLineSpan("underline-line", e), a = e.fontMetrics().defaultRuleThickness, i = F.makeVList({
      positionType: "top",
      positionData: t.height,
      children: [{
        type: "kern",
        size: a
      }, {
        type: "elem",
        elem: r
      }, {
        type: "kern",
        size: 3 * a
      }, {
        type: "elem",
        elem: t
      }]
    }, e);
    return F.makeSpan(["mord", "underline"], [i], e);
  },
  mathmlBuilder(n, e) {
    var t = new q.MathNode("mo", [new q.TextNode("‾")]);
    t.setAttribute("stretchy", "true");
    var r = new q.MathNode("munder", [pe(n.body, e), t]);
    return r.setAttribute("accentunder", "true"), r;
  }
});
H({
  type: "vcenter",
  names: ["\\vcenter"],
  props: {
    numArgs: 1,
    argTypes: ["original"],
    // In LaTeX, \vcenter can act only on a box.
    allowedInText: !1
  },
  handler(n, e) {
    var {
      parser: t
    } = n;
    return {
      type: "vcenter",
      mode: t.mode,
      body: e[0]
    };
  },
  htmlBuilder(n, e) {
    var t = le(n.body, e), r = e.fontMetrics().axisHeight, a = 0.5 * (t.height - r - (t.depth + r));
    return F.makeVList({
      positionType: "shift",
      positionData: a,
      children: [{
        type: "elem",
        elem: t
      }]
    }, e);
  },
  mathmlBuilder(n, e) {
    return new q.MathNode("mpadded", [pe(n.body, e)], ["vcenter"]);
  }
});
H({
  type: "verb",
  names: ["\\verb"],
  props: {
    numArgs: 0,
    allowedInText: !0
  },
  handler(n, e, t) {
    throw new L("\\verb ended by end of line instead of matching delimiter");
  },
  htmlBuilder(n, e) {
    for (var t = Ji(n), r = [], a = e.havingStyle(e.style.text()), i = 0; i < t.length; i++) {
      var l = t[i];
      l === "~" && (l = "\\textasciitilde"), r.push(F.makeSymbol(l, "Typewriter-Regular", n.mode, a, ["mord", "texttt"]));
    }
    return F.makeSpan(["mord", "text"].concat(a.sizingClasses(e)), F.tryCombineChars(r), a);
  },
  mathmlBuilder(n, e) {
    var t = new q.TextNode(Ji(n)), r = new q.MathNode("mtext", [t]);
    return r.setAttribute("mathvariant", "monospace"), r;
  }
});
var Ji = (n) => n.body.replace(/ /g, n.star ? "␣" : " "), c0 = Ns, po = `[ \r
	]`, Fc = "\\\\[a-zA-Z@]+", Cc = "\\\\[^\uD800-\uDFFF]", Tc = "(" + Fc + ")" + po + "*", $c = `\\\\(
|[ \r	]+
?)[ \r	]*`, oa = "[̀-ͯ]", Mc = new RegExp(oa + "+$"), zc = "(" + po + "+)|" + // whitespace
($c + "|") + // \whitespace
"([!-\\[\\]-‧‪-퟿豈-￿]" + // single codepoint
(oa + "*") + // ...plus accents
"|[\uD800-\uDBFF][\uDC00-\uDFFF]" + // surrogate pair
(oa + "*") + // ...plus accents
"|\\\\verb\\*([^]).*?\\4|\\\\verb([^*a-zA-Z]).*?\\5" + // \verb unstarred
("|" + Tc) + // \macroName + spaces
("|" + Cc + ")");
class el {
  // Category codes. The lexer only supports comment characters (14) for now.
  // MacroExpander additionally distinguishes active (13).
  constructor(e, t) {
    this.input = void 0, this.settings = void 0, this.tokenRegex = void 0, this.catcodes = void 0, this.input = e, this.settings = t, this.tokenRegex = new RegExp(zc, "g"), this.catcodes = {
      "%": 14,
      // comment character
      "~": 13
      // active character
    };
  }
  setCatcode(e, t) {
    this.catcodes[e] = t;
  }
  /**
   * This function lexes a single token.
   */
  lex() {
    var e = this.input, t = this.tokenRegex.lastIndex;
    if (t === e.length)
      return new ot("EOF", new Je(this, t, t));
    var r = this.tokenRegex.exec(e);
    if (r === null || r.index !== t)
      throw new L("Unexpected character: '" + e[t] + "'", new ot(e[t], new Je(this, t, t + 1)));
    var a = r[6] || r[3] || (r[2] ? "\\ " : " ");
    if (this.catcodes[a] === 14) {
      var i = e.indexOf(`
`, this.tokenRegex.lastIndex);
      return i === -1 ? (this.tokenRegex.lastIndex = e.length, this.settings.reportNonstrict("commentAtEnd", "% comment has no terminating newline; LaTeX would fail because of commenting the end of math mode (e.g. $)")) : this.tokenRegex.lastIndex = i + 1, this.lex();
    }
    return new ot(a, new Je(this, t, this.tokenRegex.lastIndex));
  }
}
class Bc {
  /**
   * Both arguments are optional.  The first argument is an object of
   * built-in mappings which never change.  The second argument is an object
   * of initial (global-level) mappings, which will constantly change
   * according to any global/top-level `set`s done.
   */
  constructor(e, t) {
    e === void 0 && (e = {}), t === void 0 && (t = {}), this.current = void 0, this.builtins = void 0, this.undefStack = void 0, this.current = t, this.builtins = e, this.undefStack = [];
  }
  /**
   * Start a new nested group, affecting future local `set`s.
   */
  beginGroup() {
    this.undefStack.push({});
  }
  /**
   * End current nested group, restoring values before the group began.
   */
  endGroup() {
    if (this.undefStack.length === 0)
      throw new L("Unbalanced namespace destruction: attempt to pop global namespace; please report this as a bug");
    var e = this.undefStack.pop();
    for (var t in e)
      e.hasOwnProperty(t) && (e[t] == null ? delete this.current[t] : this.current[t] = e[t]);
  }
  /**
   * Ends all currently nested groups (if any), restoring values before the
   * groups began.  Useful in case of an error in the middle of parsing.
   */
  endGroups() {
    for (; this.undefStack.length > 0; )
      this.endGroup();
  }
  /**
   * Detect whether `name` has a definition.  Equivalent to
   * `get(name) != null`.
   */
  has(e) {
    return this.current.hasOwnProperty(e) || this.builtins.hasOwnProperty(e);
  }
  /**
   * Get the current value of a name, or `undefined` if there is no value.
   *
   * Note: Do not use `if (namespace.get(...))` to detect whether a macro
   * is defined, as the definition may be the empty string which evaluates
   * to `false` in JavaScript.  Use `if (namespace.get(...) != null)` or
   * `if (namespace.has(...))`.
   */
  get(e) {
    return this.current.hasOwnProperty(e) ? this.current[e] : this.builtins[e];
  }
  /**
   * Set the current value of a name, and optionally set it globally too.
   * Local set() sets the current value and (when appropriate) adds an undo
   * operation to the undo stack.  Global set() may change the undo
   * operation at every level, so takes time linear in their number.
   * A value of undefined means to delete existing definitions.
   */
  set(e, t, r) {
    if (r === void 0 && (r = !1), r) {
      for (var a = 0; a < this.undefStack.length; a++)
        delete this.undefStack[a][e];
      this.undefStack.length > 0 && (this.undefStack[this.undefStack.length - 1][e] = t);
    } else {
      var i = this.undefStack[this.undefStack.length - 1];
      i && !i.hasOwnProperty(e) && (i[e] = this.current[e]);
    }
    t == null ? delete this.current[e] : this.current[e] = t;
  }
}
var Rc = no;
f("\\noexpand", function(n) {
  var e = n.popToken();
  return n.isExpandable(e.text) && (e.noexpand = !0, e.treatAsRelax = !0), {
    tokens: [e],
    numArgs: 0
  };
});
f("\\expandafter", function(n) {
  var e = n.popToken();
  return n.expandOnce(!0), {
    tokens: [e],
    numArgs: 0
  };
});
f("\\@firstoftwo", function(n) {
  var e = n.consumeArgs(2);
  return {
    tokens: e[0],
    numArgs: 0
  };
});
f("\\@secondoftwo", function(n) {
  var e = n.consumeArgs(2);
  return {
    tokens: e[1],
    numArgs: 0
  };
});
f("\\@ifnextchar", function(n) {
  var e = n.consumeArgs(3);
  n.consumeSpaces();
  var t = n.future();
  return e[0].length === 1 && e[0][0].text === t.text ? {
    tokens: e[1],
    numArgs: 0
  } : {
    tokens: e[2],
    numArgs: 0
  };
});
f("\\@ifstar", "\\@ifnextchar *{\\@firstoftwo{#1}}");
f("\\TextOrMath", function(n) {
  var e = n.consumeArgs(2);
  return n.mode === "text" ? {
    tokens: e[0],
    numArgs: 0
  } : {
    tokens: e[1],
    numArgs: 0
  };
});
var tl = {
  0: 0,
  1: 1,
  2: 2,
  3: 3,
  4: 4,
  5: 5,
  6: 6,
  7: 7,
  8: 8,
  9: 9,
  a: 10,
  A: 10,
  b: 11,
  B: 11,
  c: 12,
  C: 12,
  d: 13,
  D: 13,
  e: 14,
  E: 14,
  f: 15,
  F: 15
};
f("\\char", function(n) {
  var e = n.popToken(), t, r = "";
  if (e.text === "'")
    t = 8, e = n.popToken();
  else if (e.text === '"')
    t = 16, e = n.popToken();
  else if (e.text === "`")
    if (e = n.popToken(), e.text[0] === "\\")
      r = e.text.charCodeAt(1);
    else {
      if (e.text === "EOF")
        throw new L("\\char` missing argument");
      r = e.text.charCodeAt(0);
    }
  else
    t = 10;
  if (t) {
    if (r = tl[e.text], r == null || r >= t)
      throw new L("Invalid base-" + t + " digit " + e.text);
    for (var a; (a = tl[n.future().text]) != null && a < t; )
      r *= t, r += a, n.popToken();
  }
  return "\\@char{" + r + "}";
});
var Ia = (n, e, t, r) => {
  var a = n.consumeArg().tokens;
  if (a.length !== 1)
    throw new L("\\newcommand's first argument must be a macro name");
  var i = a[0].text, l = n.isDefined(i);
  if (l && !e)
    throw new L("\\newcommand{" + i + "} attempting to redefine " + (i + "; use \\renewcommand"));
  if (!l && !t)
    throw new L("\\renewcommand{" + i + "} when command " + i + " does not yet exist; use \\newcommand");
  var s = 0;
  if (a = n.consumeArg().tokens, a.length === 1 && a[0].text === "[") {
    for (var u = "", h = n.expandNextToken(); h.text !== "]" && h.text !== "EOF"; )
      u += h.text, h = n.expandNextToken();
    if (!u.match(/^\s*[0-9]+\s*$/))
      throw new L("Invalid number of arguments: " + u);
    s = parseInt(u), a = n.consumeArg().tokens;
  }
  return l && r || n.macros.set(i, {
    tokens: a,
    numArgs: s
  }), "";
};
f("\\newcommand", (n) => Ia(n, !1, !0, !1));
f("\\renewcommand", (n) => Ia(n, !0, !1, !1));
f("\\providecommand", (n) => Ia(n, !0, !0, !0));
f("\\message", (n) => {
  var e = n.consumeArgs(1)[0];
  return console.log(e.reverse().map((t) => t.text).join("")), "";
});
f("\\errmessage", (n) => {
  var e = n.consumeArgs(1)[0];
  return console.error(e.reverse().map((t) => t.text).join("")), "";
});
f("\\show", (n) => {
  var e = n.popToken(), t = e.text;
  return console.log(e, n.macros.get(t), c0[t], ge.math[t], ge.text[t]), "";
});
f("\\bgroup", "{");
f("\\egroup", "}");
f("~", "\\nobreakspace");
f("\\lq", "`");
f("\\rq", "'");
f("\\aa", "\\r a");
f("\\AA", "\\r A");
f("\\textcopyright", "\\html@mathml{\\textcircled{c}}{\\char`©}");
f("\\copyright", "\\TextOrMath{\\textcopyright}{\\text{\\textcopyright}}");
f("\\textregistered", "\\html@mathml{\\textcircled{\\scriptsize R}}{\\char`®}");
f("ℬ", "\\mathscr{B}");
f("ℰ", "\\mathscr{E}");
f("ℱ", "\\mathscr{F}");
f("ℋ", "\\mathscr{H}");
f("ℐ", "\\mathscr{I}");
f("ℒ", "\\mathscr{L}");
f("ℳ", "\\mathscr{M}");
f("ℛ", "\\mathscr{R}");
f("ℭ", "\\mathfrak{C}");
f("ℌ", "\\mathfrak{H}");
f("ℨ", "\\mathfrak{Z}");
f("\\Bbbk", "\\Bbb{k}");
f("·", "\\cdotp");
f("\\llap", "\\mathllap{\\textrm{#1}}");
f("\\rlap", "\\mathrlap{\\textrm{#1}}");
f("\\clap", "\\mathclap{\\textrm{#1}}");
f("\\mathstrut", "\\vphantom{(}");
f("\\underbar", "\\underline{\\text{#1}}");
f("\\not", '\\html@mathml{\\mathrel{\\mathrlap\\@not}}{\\char"338}');
f("\\neq", "\\html@mathml{\\mathrel{\\not=}}{\\mathrel{\\char`≠}}");
f("\\ne", "\\neq");
f("≠", "\\neq");
f("\\notin", "\\html@mathml{\\mathrel{{\\in}\\mathllap{/\\mskip1mu}}}{\\mathrel{\\char`∉}}");
f("∉", "\\notin");
f("≘", "\\html@mathml{\\mathrel{=\\kern{-1em}\\raisebox{0.4em}{$\\scriptsize\\frown$}}}{\\mathrel{\\char`≘}}");
f("≙", "\\html@mathml{\\stackrel{\\tiny\\wedge}{=}}{\\mathrel{\\char`≘}}");
f("≚", "\\html@mathml{\\stackrel{\\tiny\\vee}{=}}{\\mathrel{\\char`≚}}");
f("≛", "\\html@mathml{\\stackrel{\\scriptsize\\star}{=}}{\\mathrel{\\char`≛}}");
f("≝", "\\html@mathml{\\stackrel{\\tiny\\mathrm{def}}{=}}{\\mathrel{\\char`≝}}");
f("≞", "\\html@mathml{\\stackrel{\\tiny\\mathrm{m}}{=}}{\\mathrel{\\char`≞}}");
f("≟", "\\html@mathml{\\stackrel{\\tiny?}{=}}{\\mathrel{\\char`≟}}");
f("⟂", "\\perp");
f("‼", "\\mathclose{!\\mkern-0.8mu!}");
f("∌", "\\notni");
f("⌜", "\\ulcorner");
f("⌝", "\\urcorner");
f("⌞", "\\llcorner");
f("⌟", "\\lrcorner");
f("©", "\\copyright");
f("®", "\\textregistered");
f("️", "\\textregistered");
f("\\ulcorner", '\\html@mathml{\\@ulcorner}{\\mathop{\\char"231c}}');
f("\\urcorner", '\\html@mathml{\\@urcorner}{\\mathop{\\char"231d}}');
f("\\llcorner", '\\html@mathml{\\@llcorner}{\\mathop{\\char"231e}}');
f("\\lrcorner", '\\html@mathml{\\@lrcorner}{\\mathop{\\char"231f}}');
f("\\vdots", "{\\varvdots\\rule{0pt}{15pt}}");
f("⋮", "\\vdots");
f("\\varGamma", "\\mathit{\\Gamma}");
f("\\varDelta", "\\mathit{\\Delta}");
f("\\varTheta", "\\mathit{\\Theta}");
f("\\varLambda", "\\mathit{\\Lambda}");
f("\\varXi", "\\mathit{\\Xi}");
f("\\varPi", "\\mathit{\\Pi}");
f("\\varSigma", "\\mathit{\\Sigma}");
f("\\varUpsilon", "\\mathit{\\Upsilon}");
f("\\varPhi", "\\mathit{\\Phi}");
f("\\varPsi", "\\mathit{\\Psi}");
f("\\varOmega", "\\mathit{\\Omega}");
f("\\substack", "\\begin{subarray}{c}#1\\end{subarray}");
f("\\colon", "\\nobreak\\mskip2mu\\mathpunct{}\\mathchoice{\\mkern-3mu}{\\mkern-3mu}{}{}{:}\\mskip6mu\\relax");
f("\\boxed", "\\fbox{$\\displaystyle{#1}$}");
f("\\iff", "\\DOTSB\\;\\Longleftrightarrow\\;");
f("\\implies", "\\DOTSB\\;\\Longrightarrow\\;");
f("\\impliedby", "\\DOTSB\\;\\Longleftarrow\\;");
f("\\dddot", "{\\overset{\\raisebox{-0.1ex}{\\normalsize ...}}{#1}}");
f("\\ddddot", "{\\overset{\\raisebox{-0.1ex}{\\normalsize ....}}{#1}}");
var rl = {
  ",": "\\dotsc",
  "\\not": "\\dotsb",
  // \keybin@ checks for the following:
  "+": "\\dotsb",
  "=": "\\dotsb",
  "<": "\\dotsb",
  ">": "\\dotsb",
  "-": "\\dotsb",
  "*": "\\dotsb",
  ":": "\\dotsb",
  // Symbols whose definition starts with \DOTSB:
  "\\DOTSB": "\\dotsb",
  "\\coprod": "\\dotsb",
  "\\bigvee": "\\dotsb",
  "\\bigwedge": "\\dotsb",
  "\\biguplus": "\\dotsb",
  "\\bigcap": "\\dotsb",
  "\\bigcup": "\\dotsb",
  "\\prod": "\\dotsb",
  "\\sum": "\\dotsb",
  "\\bigotimes": "\\dotsb",
  "\\bigoplus": "\\dotsb",
  "\\bigodot": "\\dotsb",
  "\\bigsqcup": "\\dotsb",
  "\\And": "\\dotsb",
  "\\longrightarrow": "\\dotsb",
  "\\Longrightarrow": "\\dotsb",
  "\\longleftarrow": "\\dotsb",
  "\\Longleftarrow": "\\dotsb",
  "\\longleftrightarrow": "\\dotsb",
  "\\Longleftrightarrow": "\\dotsb",
  "\\mapsto": "\\dotsb",
  "\\longmapsto": "\\dotsb",
  "\\hookrightarrow": "\\dotsb",
  "\\doteq": "\\dotsb",
  // Symbols whose definition starts with \mathbin:
  "\\mathbin": "\\dotsb",
  // Symbols whose definition starts with \mathrel:
  "\\mathrel": "\\dotsb",
  "\\relbar": "\\dotsb",
  "\\Relbar": "\\dotsb",
  "\\xrightarrow": "\\dotsb",
  "\\xleftarrow": "\\dotsb",
  // Symbols whose definition starts with \DOTSI:
  "\\DOTSI": "\\dotsi",
  "\\int": "\\dotsi",
  "\\oint": "\\dotsi",
  "\\iint": "\\dotsi",
  "\\iiint": "\\dotsi",
  "\\iiiint": "\\dotsi",
  "\\idotsint": "\\dotsi",
  // Symbols whose definition starts with \DOTSX:
  "\\DOTSX": "\\dotsx"
};
f("\\dots", function(n) {
  var e = "\\dotso", t = n.expandAfterFuture().text;
  return t in rl ? e = rl[t] : (t.slice(0, 4) === "\\not" || t in ge.math && Z.contains(["bin", "rel"], ge.math[t].group)) && (e = "\\dotsb"), e;
});
var Oa = {
  // \rightdelim@ checks for the following:
  ")": !0,
  "]": !0,
  "\\rbrack": !0,
  "\\}": !0,
  "\\rbrace": !0,
  "\\rangle": !0,
  "\\rceil": !0,
  "\\rfloor": !0,
  "\\rgroup": !0,
  "\\rmoustache": !0,
  "\\right": !0,
  "\\bigr": !0,
  "\\biggr": !0,
  "\\Bigr": !0,
  "\\Biggr": !0,
  // \extra@ also tests for the following:
  $: !0,
  // \extrap@ checks for the following:
  ";": !0,
  ".": !0,
  ",": !0
};
f("\\dotso", function(n) {
  var e = n.future().text;
  return e in Oa ? "\\ldots\\," : "\\ldots";
});
f("\\dotsc", function(n) {
  var e = n.future().text;
  return e in Oa && e !== "," ? "\\ldots\\," : "\\ldots";
});
f("\\cdots", function(n) {
  var e = n.future().text;
  return e in Oa ? "\\@cdots\\," : "\\@cdots";
});
f("\\dotsb", "\\cdots");
f("\\dotsm", "\\cdots");
f("\\dotsi", "\\!\\cdots");
f("\\dotsx", "\\ldots\\,");
f("\\DOTSI", "\\relax");
f("\\DOTSB", "\\relax");
f("\\DOTSX", "\\relax");
f("\\tmspace", "\\TextOrMath{\\kern#1#3}{\\mskip#1#2}\\relax");
f("\\,", "\\tmspace+{3mu}{.1667em}");
f("\\thinspace", "\\,");
f("\\>", "\\mskip{4mu}");
f("\\:", "\\tmspace+{4mu}{.2222em}");
f("\\medspace", "\\:");
f("\\;", "\\tmspace+{5mu}{.2777em}");
f("\\thickspace", "\\;");
f("\\!", "\\tmspace-{3mu}{.1667em}");
f("\\negthinspace", "\\!");
f("\\negmedspace", "\\tmspace-{4mu}{.2222em}");
f("\\negthickspace", "\\tmspace-{5mu}{.277em}");
f("\\enspace", "\\kern.5em ");
f("\\enskip", "\\hskip.5em\\relax");
f("\\quad", "\\hskip1em\\relax");
f("\\qquad", "\\hskip2em\\relax");
f("\\tag", "\\@ifstar\\tag@literal\\tag@paren");
f("\\tag@paren", "\\tag@literal{({#1})}");
f("\\tag@literal", (n) => {
  if (n.macros.get("\\df@tag"))
    throw new L("Multiple \\tag");
  return "\\gdef\\df@tag{\\text{#1}}";
});
f("\\bmod", "\\mathchoice{\\mskip1mu}{\\mskip1mu}{\\mskip5mu}{\\mskip5mu}\\mathbin{\\rm mod}\\mathchoice{\\mskip1mu}{\\mskip1mu}{\\mskip5mu}{\\mskip5mu}");
f("\\pod", "\\allowbreak\\mathchoice{\\mkern18mu}{\\mkern8mu}{\\mkern8mu}{\\mkern8mu}(#1)");
f("\\pmod", "\\pod{{\\rm mod}\\mkern6mu#1}");
f("\\mod", "\\allowbreak\\mathchoice{\\mkern18mu}{\\mkern12mu}{\\mkern12mu}{\\mkern12mu}{\\rm mod}\\,\\,#1");
f("\\newline", "\\\\\\relax");
f("\\TeX", "\\textrm{\\html@mathml{T\\kern-.1667em\\raisebox{-.5ex}{E}\\kern-.125emX}{TeX}}");
var go = P(zt["Main-Regular"][84][1] - 0.7 * zt["Main-Regular"][65][1]);
f("\\LaTeX", "\\textrm{\\html@mathml{" + ("L\\kern-.36em\\raisebox{" + go + "}{\\scriptstyle A}") + "\\kern-.15em\\TeX}{LaTeX}}");
f("\\KaTeX", "\\textrm{\\html@mathml{" + ("K\\kern-.17em\\raisebox{" + go + "}{\\scriptstyle A}") + "\\kern-.15em\\TeX}{KaTeX}}");
f("\\hspace", "\\@ifstar\\@hspacer\\@hspace");
f("\\@hspace", "\\hskip #1\\relax");
f("\\@hspacer", "\\rule{0pt}{0pt}\\hskip #1\\relax");
f("\\ordinarycolon", ":");
f("\\vcentcolon", "\\mathrel{\\mathop\\ordinarycolon}");
f("\\dblcolon", '\\html@mathml{\\mathrel{\\vcentcolon\\mathrel{\\mkern-.9mu}\\vcentcolon}}{\\mathop{\\char"2237}}');
f("\\coloneqq", '\\html@mathml{\\mathrel{\\vcentcolon\\mathrel{\\mkern-1.2mu}=}}{\\mathop{\\char"2254}}');
f("\\Coloneqq", '\\html@mathml{\\mathrel{\\dblcolon\\mathrel{\\mkern-1.2mu}=}}{\\mathop{\\char"2237\\char"3d}}');
f("\\coloneq", '\\html@mathml{\\mathrel{\\vcentcolon\\mathrel{\\mkern-1.2mu}\\mathrel{-}}}{\\mathop{\\char"3a\\char"2212}}');
f("\\Coloneq", '\\html@mathml{\\mathrel{\\dblcolon\\mathrel{\\mkern-1.2mu}\\mathrel{-}}}{\\mathop{\\char"2237\\char"2212}}');
f("\\eqqcolon", '\\html@mathml{\\mathrel{=\\mathrel{\\mkern-1.2mu}\\vcentcolon}}{\\mathop{\\char"2255}}');
f("\\Eqqcolon", '\\html@mathml{\\mathrel{=\\mathrel{\\mkern-1.2mu}\\dblcolon}}{\\mathop{\\char"3d\\char"2237}}');
f("\\eqcolon", '\\html@mathml{\\mathrel{\\mathrel{-}\\mathrel{\\mkern-1.2mu}\\vcentcolon}}{\\mathop{\\char"2239}}');
f("\\Eqcolon", '\\html@mathml{\\mathrel{\\mathrel{-}\\mathrel{\\mkern-1.2mu}\\dblcolon}}{\\mathop{\\char"2212\\char"2237}}');
f("\\colonapprox", '\\html@mathml{\\mathrel{\\vcentcolon\\mathrel{\\mkern-1.2mu}\\approx}}{\\mathop{\\char"3a\\char"2248}}');
f("\\Colonapprox", '\\html@mathml{\\mathrel{\\dblcolon\\mathrel{\\mkern-1.2mu}\\approx}}{\\mathop{\\char"2237\\char"2248}}');
f("\\colonsim", '\\html@mathml{\\mathrel{\\vcentcolon\\mathrel{\\mkern-1.2mu}\\sim}}{\\mathop{\\char"3a\\char"223c}}');
f("\\Colonsim", '\\html@mathml{\\mathrel{\\dblcolon\\mathrel{\\mkern-1.2mu}\\sim}}{\\mathop{\\char"2237\\char"223c}}');
f("∷", "\\dblcolon");
f("∹", "\\eqcolon");
f("≔", "\\coloneqq");
f("≕", "\\eqqcolon");
f("⩴", "\\Coloneqq");
f("\\ratio", "\\vcentcolon");
f("\\coloncolon", "\\dblcolon");
f("\\colonequals", "\\coloneqq");
f("\\coloncolonequals", "\\Coloneqq");
f("\\equalscolon", "\\eqqcolon");
f("\\equalscoloncolon", "\\Eqqcolon");
f("\\colonminus", "\\coloneq");
f("\\coloncolonminus", "\\Coloneq");
f("\\minuscolon", "\\eqcolon");
f("\\minuscoloncolon", "\\Eqcolon");
f("\\coloncolonapprox", "\\Colonapprox");
f("\\coloncolonsim", "\\Colonsim");
f("\\simcolon", "\\mathrel{\\sim\\mathrel{\\mkern-1.2mu}\\vcentcolon}");
f("\\simcoloncolon", "\\mathrel{\\sim\\mathrel{\\mkern-1.2mu}\\dblcolon}");
f("\\approxcolon", "\\mathrel{\\approx\\mathrel{\\mkern-1.2mu}\\vcentcolon}");
f("\\approxcoloncolon", "\\mathrel{\\approx\\mathrel{\\mkern-1.2mu}\\dblcolon}");
f("\\notni", "\\html@mathml{\\not\\ni}{\\mathrel{\\char`∌}}");
f("\\limsup", "\\DOTSB\\operatorname*{lim\\,sup}");
f("\\liminf", "\\DOTSB\\operatorname*{lim\\,inf}");
f("\\injlim", "\\DOTSB\\operatorname*{inj\\,lim}");
f("\\projlim", "\\DOTSB\\operatorname*{proj\\,lim}");
f("\\varlimsup", "\\DOTSB\\operatorname*{\\overline{lim}}");
f("\\varliminf", "\\DOTSB\\operatorname*{\\underline{lim}}");
f("\\varinjlim", "\\DOTSB\\operatorname*{\\underrightarrow{lim}}");
f("\\varprojlim", "\\DOTSB\\operatorname*{\\underleftarrow{lim}}");
f("\\gvertneqq", "\\html@mathml{\\@gvertneqq}{≩}");
f("\\lvertneqq", "\\html@mathml{\\@lvertneqq}{≨}");
f("\\ngeqq", "\\html@mathml{\\@ngeqq}{≱}");
f("\\ngeqslant", "\\html@mathml{\\@ngeqslant}{≱}");
f("\\nleqq", "\\html@mathml{\\@nleqq}{≰}");
f("\\nleqslant", "\\html@mathml{\\@nleqslant}{≰}");
f("\\nshortmid", "\\html@mathml{\\@nshortmid}{∤}");
f("\\nshortparallel", "\\html@mathml{\\@nshortparallel}{∦}");
f("\\nsubseteqq", "\\html@mathml{\\@nsubseteqq}{⊈}");
f("\\nsupseteqq", "\\html@mathml{\\@nsupseteqq}{⊉}");
f("\\varsubsetneq", "\\html@mathml{\\@varsubsetneq}{⊊}");
f("\\varsubsetneqq", "\\html@mathml{\\@varsubsetneqq}{⫋}");
f("\\varsupsetneq", "\\html@mathml{\\@varsupsetneq}{⊋}");
f("\\varsupsetneqq", "\\html@mathml{\\@varsupsetneqq}{⫌}");
f("\\imath", "\\html@mathml{\\@imath}{ı}");
f("\\jmath", "\\html@mathml{\\@jmath}{ȷ}");
f("\\llbracket", "\\html@mathml{\\mathopen{[\\mkern-3.2mu[}}{\\mathopen{\\char`⟦}}");
f("\\rrbracket", "\\html@mathml{\\mathclose{]\\mkern-3.2mu]}}{\\mathclose{\\char`⟧}}");
f("⟦", "\\llbracket");
f("⟧", "\\rrbracket");
f("\\lBrace", "\\html@mathml{\\mathopen{\\{\\mkern-3.2mu[}}{\\mathopen{\\char`⦃}}");
f("\\rBrace", "\\html@mathml{\\mathclose{]\\mkern-3.2mu\\}}}{\\mathclose{\\char`⦄}}");
f("⦃", "\\lBrace");
f("⦄", "\\rBrace");
f("\\minuso", "\\mathbin{\\html@mathml{{\\mathrlap{\\mathchoice{\\kern{0.145em}}{\\kern{0.145em}}{\\kern{0.1015em}}{\\kern{0.0725em}}\\circ}{-}}}{\\char`⦵}}");
f("⦵", "\\minuso");
f("\\darr", "\\downarrow");
f("\\dArr", "\\Downarrow");
f("\\Darr", "\\Downarrow");
f("\\lang", "\\langle");
f("\\rang", "\\rangle");
f("\\uarr", "\\uparrow");
f("\\uArr", "\\Uparrow");
f("\\Uarr", "\\Uparrow");
f("\\N", "\\mathbb{N}");
f("\\R", "\\mathbb{R}");
f("\\Z", "\\mathbb{Z}");
f("\\alef", "\\aleph");
f("\\alefsym", "\\aleph");
f("\\Alpha", "\\mathrm{A}");
f("\\Beta", "\\mathrm{B}");
f("\\bull", "\\bullet");
f("\\Chi", "\\mathrm{X}");
f("\\clubs", "\\clubsuit");
f("\\cnums", "\\mathbb{C}");
f("\\Complex", "\\mathbb{C}");
f("\\Dagger", "\\ddagger");
f("\\diamonds", "\\diamondsuit");
f("\\empty", "\\emptyset");
f("\\Epsilon", "\\mathrm{E}");
f("\\Eta", "\\mathrm{H}");
f("\\exist", "\\exists");
f("\\harr", "\\leftrightarrow");
f("\\hArr", "\\Leftrightarrow");
f("\\Harr", "\\Leftrightarrow");
f("\\hearts", "\\heartsuit");
f("\\image", "\\Im");
f("\\infin", "\\infty");
f("\\Iota", "\\mathrm{I}");
f("\\isin", "\\in");
f("\\Kappa", "\\mathrm{K}");
f("\\larr", "\\leftarrow");
f("\\lArr", "\\Leftarrow");
f("\\Larr", "\\Leftarrow");
f("\\lrarr", "\\leftrightarrow");
f("\\lrArr", "\\Leftrightarrow");
f("\\Lrarr", "\\Leftrightarrow");
f("\\Mu", "\\mathrm{M}");
f("\\natnums", "\\mathbb{N}");
f("\\Nu", "\\mathrm{N}");
f("\\Omicron", "\\mathrm{O}");
f("\\plusmn", "\\pm");
f("\\rarr", "\\rightarrow");
f("\\rArr", "\\Rightarrow");
f("\\Rarr", "\\Rightarrow");
f("\\real", "\\Re");
f("\\reals", "\\mathbb{R}");
f("\\Reals", "\\mathbb{R}");
f("\\Rho", "\\mathrm{P}");
f("\\sdot", "\\cdot");
f("\\sect", "\\S");
f("\\spades", "\\spadesuit");
f("\\sub", "\\subset");
f("\\sube", "\\subseteq");
f("\\supe", "\\supseteq");
f("\\Tau", "\\mathrm{T}");
f("\\thetasym", "\\vartheta");
f("\\weierp", "\\wp");
f("\\Zeta", "\\mathrm{Z}");
f("\\argmin", "\\DOTSB\\operatorname*{arg\\,min}");
f("\\argmax", "\\DOTSB\\operatorname*{arg\\,max}");
f("\\plim", "\\DOTSB\\mathop{\\operatorname{plim}}\\limits");
f("\\bra", "\\mathinner{\\langle{#1}|}");
f("\\ket", "\\mathinner{|{#1}\\rangle}");
f("\\braket", "\\mathinner{\\langle{#1}\\rangle}");
f("\\Bra", "\\left\\langle#1\\right|");
f("\\Ket", "\\left|#1\\right\\rangle");
var vo = (n) => (e) => {
  var t = e.consumeArg().tokens, r = e.consumeArg().tokens, a = e.consumeArg().tokens, i = e.consumeArg().tokens, l = e.macros.get("|"), s = e.macros.get("\\|");
  e.macros.beginGroup();
  var u = (g) => (p) => {
    n && (p.macros.set("|", l), a.length && p.macros.set("\\|", s));
    var v = g;
    if (!g && a.length) {
      var k = p.future();
      k.text === "|" && (p.popToken(), v = !0);
    }
    return {
      tokens: v ? a : r,
      numArgs: 0
    };
  };
  e.macros.set("|", u(!1)), a.length && e.macros.set("\\|", u(!0));
  var h = e.consumeArg().tokens, d = e.expandTokens([
    ...i,
    ...h,
    ...t
    // reversed
  ]);
  return e.macros.endGroup(), {
    tokens: d.reverse(),
    numArgs: 0
  };
};
f("\\bra@ket", vo(!1));
f("\\bra@set", vo(!0));
f("\\Braket", "\\bra@ket{\\left\\langle}{\\,\\middle\\vert\\,}{\\,\\middle\\vert\\,}{\\right\\rangle}");
f("\\Set", "\\bra@set{\\left\\{\\:}{\\;\\middle\\vert\\;}{\\;\\middle\\Vert\\;}{\\:\\right\\}}");
f("\\set", "\\bra@set{\\{\\,}{\\mid}{}{\\,\\}}");
f("\\angln", "{\\angl n}");
f("\\blue", "\\textcolor{##6495ed}{#1}");
f("\\orange", "\\textcolor{##ffa500}{#1}");
f("\\pink", "\\textcolor{##ff00af}{#1}");
f("\\red", "\\textcolor{##df0030}{#1}");
f("\\green", "\\textcolor{##28ae7b}{#1}");
f("\\gray", "\\textcolor{gray}{#1}");
f("\\purple", "\\textcolor{##9d38bd}{#1}");
f("\\blueA", "\\textcolor{##ccfaff}{#1}");
f("\\blueB", "\\textcolor{##80f6ff}{#1}");
f("\\blueC", "\\textcolor{##63d9ea}{#1}");
f("\\blueD", "\\textcolor{##11accd}{#1}");
f("\\blueE", "\\textcolor{##0c7f99}{#1}");
f("\\tealA", "\\textcolor{##94fff5}{#1}");
f("\\tealB", "\\textcolor{##26edd5}{#1}");
f("\\tealC", "\\textcolor{##01d1c1}{#1}");
f("\\tealD", "\\textcolor{##01a995}{#1}");
f("\\tealE", "\\textcolor{##208170}{#1}");
f("\\greenA", "\\textcolor{##b6ffb0}{#1}");
f("\\greenB", "\\textcolor{##8af281}{#1}");
f("\\greenC", "\\textcolor{##74cf70}{#1}");
f("\\greenD", "\\textcolor{##1fab54}{#1}");
f("\\greenE", "\\textcolor{##0d923f}{#1}");
f("\\goldA", "\\textcolor{##ffd0a9}{#1}");
f("\\goldB", "\\textcolor{##ffbb71}{#1}");
f("\\goldC", "\\textcolor{##ff9c39}{#1}");
f("\\goldD", "\\textcolor{##e07d10}{#1}");
f("\\goldE", "\\textcolor{##a75a05}{#1}");
f("\\redA", "\\textcolor{##fca9a9}{#1}");
f("\\redB", "\\textcolor{##ff8482}{#1}");
f("\\redC", "\\textcolor{##f9685d}{#1}");
f("\\redD", "\\textcolor{##e84d39}{#1}");
f("\\redE", "\\textcolor{##bc2612}{#1}");
f("\\maroonA", "\\textcolor{##ffbde0}{#1}");
f("\\maroonB", "\\textcolor{##ff92c6}{#1}");
f("\\maroonC", "\\textcolor{##ed5fa6}{#1}");
f("\\maroonD", "\\textcolor{##ca337c}{#1}");
f("\\maroonE", "\\textcolor{##9e034e}{#1}");
f("\\purpleA", "\\textcolor{##ddd7ff}{#1}");
f("\\purpleB", "\\textcolor{##c6b9fc}{#1}");
f("\\purpleC", "\\textcolor{##aa87ff}{#1}");
f("\\purpleD", "\\textcolor{##7854ab}{#1}");
f("\\purpleE", "\\textcolor{##543b78}{#1}");
f("\\mintA", "\\textcolor{##f5f9e8}{#1}");
f("\\mintB", "\\textcolor{##edf2df}{#1}");
f("\\mintC", "\\textcolor{##e0e5cc}{#1}");
f("\\grayA", "\\textcolor{##f6f7f7}{#1}");
f("\\grayB", "\\textcolor{##f0f1f2}{#1}");
f("\\grayC", "\\textcolor{##e3e5e6}{#1}");
f("\\grayD", "\\textcolor{##d6d8da}{#1}");
f("\\grayE", "\\textcolor{##babec2}{#1}");
f("\\grayF", "\\textcolor{##888d93}{#1}");
f("\\grayG", "\\textcolor{##626569}{#1}");
f("\\grayH", "\\textcolor{##3b3e40}{#1}");
f("\\grayI", "\\textcolor{##21242c}{#1}");
f("\\kaBlue", "\\textcolor{##314453}{#1}");
f("\\kaGreen", "\\textcolor{##71B307}{#1}");
var _o = {
  "^": !0,
  // Parser.js
  _: !0,
  // Parser.js
  "\\limits": !0,
  // Parser.js
  "\\nolimits": !0
  // Parser.js
};
class Nc {
  constructor(e, t, r) {
    this.settings = void 0, this.expansionCount = void 0, this.lexer = void 0, this.macros = void 0, this.stack = void 0, this.mode = void 0, this.settings = t, this.expansionCount = 0, this.feed(e), this.macros = new Bc(Rc, t.macros), this.mode = r, this.stack = [];
  }
  /**
   * Feed a new input string to the same MacroExpander
   * (with existing macros etc.).
   */
  feed(e) {
    this.lexer = new el(e, this.settings);
  }
  /**
   * Switches between "text" and "math" modes.
   */
  switchMode(e) {
    this.mode = e;
  }
  /**
   * Start a new group nesting within all namespaces.
   */
  beginGroup() {
    this.macros.beginGroup();
  }
  /**
   * End current group nesting within all namespaces.
   */
  endGroup() {
    this.macros.endGroup();
  }
  /**
   * Ends all currently nested groups (if any), restoring values before the
   * groups began.  Useful in case of an error in the middle of parsing.
   */
  endGroups() {
    this.macros.endGroups();
  }
  /**
   * Returns the topmost token on the stack, without expanding it.
   * Similar in behavior to TeX's `\futurelet`.
   */
  future() {
    return this.stack.length === 0 && this.pushToken(this.lexer.lex()), this.stack[this.stack.length - 1];
  }
  /**
   * Remove and return the next unexpanded token.
   */
  popToken() {
    return this.future(), this.stack.pop();
  }
  /**
   * Add a given token to the token stack.  In particular, this get be used
   * to put back a token returned from one of the other methods.
   */
  pushToken(e) {
    this.stack.push(e);
  }
  /**
   * Append an array of tokens to the token stack.
   */
  pushTokens(e) {
    this.stack.push(...e);
  }
  /**
   * Find an macro argument without expanding tokens and append the array of
   * tokens to the token stack. Uses Token as a container for the result.
   */
  scanArgument(e) {
    var t, r, a;
    if (e) {
      if (this.consumeSpaces(), this.future().text !== "[")
        return null;
      t = this.popToken(), {
        tokens: a,
        end: r
      } = this.consumeArg(["]"]);
    } else
      ({
        tokens: a,
        start: t,
        end: r
      } = this.consumeArg());
    return this.pushToken(new ot("EOF", r.loc)), this.pushTokens(a), t.range(r, "");
  }
  /**
   * Consume all following space tokens, without expansion.
   */
  consumeSpaces() {
    for (; ; ) {
      var e = this.future();
      if (e.text === " ")
        this.stack.pop();
      else
        break;
    }
  }
  /**
   * Consume an argument from the token stream, and return the resulting array
   * of tokens and start/end token.
   */
  consumeArg(e) {
    var t = [], r = e && e.length > 0;
    r || this.consumeSpaces();
    var a = this.future(), i, l = 0, s = 0;
    do {
      if (i = this.popToken(), t.push(i), i.text === "{")
        ++l;
      else if (i.text === "}") {
        if (--l, l === -1)
          throw new L("Extra }", i);
      } else if (i.text === "EOF")
        throw new L("Unexpected end of input in a macro argument, expected '" + (e && r ? e[s] : "}") + "'", i);
      if (e && r)
        if ((l === 0 || l === 1 && e[s] === "{") && i.text === e[s]) {
          if (++s, s === e.length) {
            t.splice(-s, s);
            break;
          }
        } else
          s = 0;
    } while (l !== 0 || r);
    return a.text === "{" && t[t.length - 1].text === "}" && (t.pop(), t.shift()), t.reverse(), {
      tokens: t,
      start: a,
      end: i
    };
  }
  /**
   * Consume the specified number of (delimited) arguments from the token
   * stream and return the resulting array of arguments.
   */
  consumeArgs(e, t) {
    if (t) {
      if (t.length !== e + 1)
        throw new L("The length of delimiters doesn't match the number of args!");
      for (var r = t[0], a = 0; a < r.length; a++) {
        var i = this.popToken();
        if (r[a] !== i.text)
          throw new L("Use of the macro doesn't match its definition", i);
      }
    }
    for (var l = [], s = 0; s < e; s++)
      l.push(this.consumeArg(t && t[s + 1]).tokens);
    return l;
  }
  /**
   * Increment `expansionCount` by the specified amount.
   * Throw an error if it exceeds `maxExpand`.
   */
  countExpansion(e) {
    if (this.expansionCount += e, this.expansionCount > this.settings.maxExpand)
      throw new L("Too many expansions: infinite loop or need to increase maxExpand setting");
  }
  /**
   * Expand the next token only once if possible.
   *
   * If the token is expanded, the resulting tokens will be pushed onto
   * the stack in reverse order, and the number of such tokens will be
   * returned.  This number might be zero or positive.
   *
   * If not, the return value is `false`, and the next token remains at the
   * top of the stack.
   *
   * In either case, the next token will be on the top of the stack,
   * or the stack will be empty (in case of empty expansion
   * and no other tokens).
   *
   * Used to implement `expandAfterFuture` and `expandNextToken`.
   *
   * If expandableOnly, only expandable tokens are expanded and
   * an undefined control sequence results in an error.
   */
  expandOnce(e) {
    var t = this.popToken(), r = t.text, a = t.noexpand ? null : this._getExpansion(r);
    if (a == null || e && a.unexpandable) {
      if (e && a == null && r[0] === "\\" && !this.isDefined(r))
        throw new L("Undefined control sequence: " + r);
      return this.pushToken(t), !1;
    }
    this.countExpansion(1);
    var i = a.tokens, l = this.consumeArgs(a.numArgs, a.delimiters);
    if (a.numArgs) {
      i = i.slice();
      for (var s = i.length - 1; s >= 0; --s) {
        var u = i[s];
        if (u.text === "#") {
          if (s === 0)
            throw new L("Incomplete placeholder at end of macro body", u);
          if (u = i[--s], u.text === "#")
            i.splice(s + 1, 1);
          else if (/^[1-9]$/.test(u.text))
            i.splice(s, 2, ...l[+u.text - 1]);
          else
            throw new L("Not a valid argument number", u);
        }
      }
    }
    return this.pushTokens(i), i.length;
  }
  /**
   * Expand the next token only once (if possible), and return the resulting
   * top token on the stack (without removing anything from the stack).
   * Similar in behavior to TeX's `\expandafter\futurelet`.
   * Equivalent to expandOnce() followed by future().
   */
  expandAfterFuture() {
    return this.expandOnce(), this.future();
  }
  /**
   * Recursively expand first token, then return first non-expandable token.
   */
  expandNextToken() {
    for (; ; )
      if (this.expandOnce() === !1) {
        var e = this.stack.pop();
        return e.treatAsRelax && (e.text = "\\relax"), e;
      }
    throw new Error();
  }
  /**
   * Fully expand the given macro name and return the resulting list of
   * tokens, or return `undefined` if no such macro is defined.
   */
  expandMacro(e) {
    return this.macros.has(e) ? this.expandTokens([new ot(e)]) : void 0;
  }
  /**
   * Fully expand the given token stream and return the resulting list of
   * tokens.  Note that the input tokens are in reverse order, but the
   * output tokens are in forward order.
   */
  expandTokens(e) {
    var t = [], r = this.stack.length;
    for (this.pushTokens(e); this.stack.length > r; )
      if (this.expandOnce(!0) === !1) {
        var a = this.stack.pop();
        a.treatAsRelax && (a.noexpand = !1, a.treatAsRelax = !1), t.push(a);
      }
    return this.countExpansion(t.length), t;
  }
  /**
   * Fully expand the given macro name and return the result as a string,
   * or return `undefined` if no such macro is defined.
   */
  expandMacroAsText(e) {
    var t = this.expandMacro(e);
    return t && t.map((r) => r.text).join("");
  }
  /**
   * Returns the expanded macro as a reversed array of tokens and a macro
   * argument count.  Or returns `null` if no such macro.
   */
  _getExpansion(e) {
    var t = this.macros.get(e);
    if (t == null)
      return t;
    if (e.length === 1) {
      var r = this.lexer.catcodes[e];
      if (r != null && r !== 13)
        return;
    }
    var a = typeof t == "function" ? t(this) : t;
    if (typeof a == "string") {
      var i = 0;
      if (a.indexOf("#") !== -1)
        for (var l = a.replace(/##/g, ""); l.indexOf("#" + (i + 1)) !== -1; )
          ++i;
      for (var s = new el(a, this.settings), u = [], h = s.lex(); h.text !== "EOF"; )
        u.push(h), h = s.lex();
      u.reverse();
      var d = {
        tokens: u,
        numArgs: i
      };
      return d;
    }
    return a;
  }
  /**
   * Determine whether a command is currently "defined" (has some
   * functionality), meaning that it's a macro (in the current group),
   * a function, a symbol, or one of the special commands listed in
   * `implicitCommands`.
   */
  isDefined(e) {
    return this.macros.has(e) || c0.hasOwnProperty(e) || ge.math.hasOwnProperty(e) || ge.text.hasOwnProperty(e) || _o.hasOwnProperty(e);
  }
  /**
   * Determine whether a command is expandable.
   */
  isExpandable(e) {
    var t = this.macros.get(e);
    return t != null ? typeof t == "string" || typeof t == "function" || !t.unexpandable : c0.hasOwnProperty(e) && !c0[e].primitive;
  }
}
var nl = /^[₊₋₌₍₎₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵦᵧᵨᵩᵪ]/, Er = Object.freeze({
  "₊": "+",
  "₋": "-",
  "₌": "=",
  "₍": "(",
  "₎": ")",
  "₀": "0",
  "₁": "1",
  "₂": "2",
  "₃": "3",
  "₄": "4",
  "₅": "5",
  "₆": "6",
  "₇": "7",
  "₈": "8",
  "₉": "9",
  "ₐ": "a",
  "ₑ": "e",
  "ₕ": "h",
  "ᵢ": "i",
  "ⱼ": "j",
  "ₖ": "k",
  "ₗ": "l",
  "ₘ": "m",
  "ₙ": "n",
  "ₒ": "o",
  "ₚ": "p",
  "ᵣ": "r",
  "ₛ": "s",
  "ₜ": "t",
  "ᵤ": "u",
  "ᵥ": "v",
  "ₓ": "x",
  "ᵦ": "β",
  "ᵧ": "γ",
  "ᵨ": "ρ",
  "ᵩ": "ϕ",
  "ᵪ": "χ",
  "⁺": "+",
  "⁻": "-",
  "⁼": "=",
  "⁽": "(",
  "⁾": ")",
  "⁰": "0",
  "¹": "1",
  "²": "2",
  "³": "3",
  "⁴": "4",
  "⁵": "5",
  "⁶": "6",
  "⁷": "7",
  "⁸": "8",
  "⁹": "9",
  "ᴬ": "A",
  "ᴮ": "B",
  "ᴰ": "D",
  "ᴱ": "E",
  "ᴳ": "G",
  "ᴴ": "H",
  "ᴵ": "I",
  "ᴶ": "J",
  "ᴷ": "K",
  "ᴸ": "L",
  "ᴹ": "M",
  "ᴺ": "N",
  "ᴼ": "O",
  "ᴾ": "P",
  "ᴿ": "R",
  "ᵀ": "T",
  "ᵁ": "U",
  "ⱽ": "V",
  "ᵂ": "W",
  "ᵃ": "a",
  "ᵇ": "b",
  "ᶜ": "c",
  "ᵈ": "d",
  "ᵉ": "e",
  "ᶠ": "f",
  "ᵍ": "g",
  ʰ: "h",
  "ⁱ": "i",
  ʲ: "j",
  "ᵏ": "k",
  ˡ: "l",
  "ᵐ": "m",
  ⁿ: "n",
  "ᵒ": "o",
  "ᵖ": "p",
  ʳ: "r",
  ˢ: "s",
  "ᵗ": "t",
  "ᵘ": "u",
  "ᵛ": "v",
  ʷ: "w",
  ˣ: "x",
  ʸ: "y",
  "ᶻ": "z",
  "ᵝ": "β",
  "ᵞ": "γ",
  "ᵟ": "δ",
  "ᵠ": "ϕ",
  "ᵡ": "χ",
  "ᶿ": "θ"
}), Ln = {
  "́": {
    text: "\\'",
    math: "\\acute"
  },
  "̀": {
    text: "\\`",
    math: "\\grave"
  },
  "̈": {
    text: '\\"',
    math: "\\ddot"
  },
  "̃": {
    text: "\\~",
    math: "\\tilde"
  },
  "̄": {
    text: "\\=",
    math: "\\bar"
  },
  "̆": {
    text: "\\u",
    math: "\\breve"
  },
  "̌": {
    text: "\\v",
    math: "\\check"
  },
  "̂": {
    text: "\\^",
    math: "\\hat"
  },
  "̇": {
    text: "\\.",
    math: "\\dot"
  },
  "̊": {
    text: "\\r",
    math: "\\mathring"
  },
  "̋": {
    text: "\\H"
  },
  "̧": {
    text: "\\c"
  }
}, al = {
  á: "á",
  à: "à",
  ä: "ä",
  ǟ: "ǟ",
  ã: "ã",
  ā: "ā",
  ă: "ă",
  ắ: "ắ",
  ằ: "ằ",
  ẵ: "ẵ",
  ǎ: "ǎ",
  â: "â",
  ấ: "ấ",
  ầ: "ầ",
  ẫ: "ẫ",
  ȧ: "ȧ",
  ǡ: "ǡ",
  å: "å",
  ǻ: "ǻ",
  ḃ: "ḃ",
  ć: "ć",
  ḉ: "ḉ",
  č: "č",
  ĉ: "ĉ",
  ċ: "ċ",
  ç: "ç",
  ď: "ď",
  ḋ: "ḋ",
  ḑ: "ḑ",
  é: "é",
  è: "è",
  ë: "ë",
  ẽ: "ẽ",
  ē: "ē",
  ḗ: "ḗ",
  ḕ: "ḕ",
  ĕ: "ĕ",
  ḝ: "ḝ",
  ě: "ě",
  ê: "ê",
  ế: "ế",
  ề: "ề",
  ễ: "ễ",
  ė: "ė",
  ȩ: "ȩ",
  ḟ: "ḟ",
  ǵ: "ǵ",
  ḡ: "ḡ",
  ğ: "ğ",
  ǧ: "ǧ",
  ĝ: "ĝ",
  ġ: "ġ",
  ģ: "ģ",
  ḧ: "ḧ",
  ȟ: "ȟ",
  ĥ: "ĥ",
  ḣ: "ḣ",
  ḩ: "ḩ",
  í: "í",
  ì: "ì",
  ï: "ï",
  ḯ: "ḯ",
  ĩ: "ĩ",
  ī: "ī",
  ĭ: "ĭ",
  ǐ: "ǐ",
  î: "î",
  ǰ: "ǰ",
  ĵ: "ĵ",
  ḱ: "ḱ",
  ǩ: "ǩ",
  ķ: "ķ",
  ĺ: "ĺ",
  ľ: "ľ",
  ļ: "ļ",
  ḿ: "ḿ",
  ṁ: "ṁ",
  ń: "ń",
  ǹ: "ǹ",
  ñ: "ñ",
  ň: "ň",
  ṅ: "ṅ",
  ņ: "ņ",
  ó: "ó",
  ò: "ò",
  ö: "ö",
  ȫ: "ȫ",
  õ: "õ",
  ṍ: "ṍ",
  ṏ: "ṏ",
  ȭ: "ȭ",
  ō: "ō",
  ṓ: "ṓ",
  ṑ: "ṑ",
  ŏ: "ŏ",
  ǒ: "ǒ",
  ô: "ô",
  ố: "ố",
  ồ: "ồ",
  ỗ: "ỗ",
  ȯ: "ȯ",
  ȱ: "ȱ",
  ő: "ő",
  ṕ: "ṕ",
  ṗ: "ṗ",
  ŕ: "ŕ",
  ř: "ř",
  ṙ: "ṙ",
  ŗ: "ŗ",
  ś: "ś",
  ṥ: "ṥ",
  š: "š",
  ṧ: "ṧ",
  ŝ: "ŝ",
  ṡ: "ṡ",
  ş: "ş",
  ẗ: "ẗ",
  ť: "ť",
  ṫ: "ṫ",
  ţ: "ţ",
  ú: "ú",
  ù: "ù",
  ü: "ü",
  ǘ: "ǘ",
  ǜ: "ǜ",
  ǖ: "ǖ",
  ǚ: "ǚ",
  ũ: "ũ",
  ṹ: "ṹ",
  ū: "ū",
  ṻ: "ṻ",
  ŭ: "ŭ",
  ǔ: "ǔ",
  û: "û",
  ů: "ů",
  ű: "ű",
  ṽ: "ṽ",
  ẃ: "ẃ",
  ẁ: "ẁ",
  ẅ: "ẅ",
  ŵ: "ŵ",
  ẇ: "ẇ",
  ẘ: "ẘ",
  ẍ: "ẍ",
  ẋ: "ẋ",
  ý: "ý",
  ỳ: "ỳ",
  ÿ: "ÿ",
  ỹ: "ỹ",
  ȳ: "ȳ",
  ŷ: "ŷ",
  ẏ: "ẏ",
  ẙ: "ẙ",
  ź: "ź",
  ž: "ž",
  ẑ: "ẑ",
  ż: "ż",
  Á: "Á",
  À: "À",
  Ä: "Ä",
  Ǟ: "Ǟ",
  Ã: "Ã",
  Ā: "Ā",
  Ă: "Ă",
  Ắ: "Ắ",
  Ằ: "Ằ",
  Ẵ: "Ẵ",
  Ǎ: "Ǎ",
  Â: "Â",
  Ấ: "Ấ",
  Ầ: "Ầ",
  Ẫ: "Ẫ",
  Ȧ: "Ȧ",
  Ǡ: "Ǡ",
  Å: "Å",
  Ǻ: "Ǻ",
  Ḃ: "Ḃ",
  Ć: "Ć",
  Ḉ: "Ḉ",
  Č: "Č",
  Ĉ: "Ĉ",
  Ċ: "Ċ",
  Ç: "Ç",
  Ď: "Ď",
  Ḋ: "Ḋ",
  Ḑ: "Ḑ",
  É: "É",
  È: "È",
  Ë: "Ë",
  Ẽ: "Ẽ",
  Ē: "Ē",
  Ḗ: "Ḗ",
  Ḕ: "Ḕ",
  Ĕ: "Ĕ",
  Ḝ: "Ḝ",
  Ě: "Ě",
  Ê: "Ê",
  Ế: "Ế",
  Ề: "Ề",
  Ễ: "Ễ",
  Ė: "Ė",
  Ȩ: "Ȩ",
  Ḟ: "Ḟ",
  Ǵ: "Ǵ",
  Ḡ: "Ḡ",
  Ğ: "Ğ",
  Ǧ: "Ǧ",
  Ĝ: "Ĝ",
  Ġ: "Ġ",
  Ģ: "Ģ",
  Ḧ: "Ḧ",
  Ȟ: "Ȟ",
  Ĥ: "Ĥ",
  Ḣ: "Ḣ",
  Ḩ: "Ḩ",
  Í: "Í",
  Ì: "Ì",
  Ï: "Ï",
  Ḯ: "Ḯ",
  Ĩ: "Ĩ",
  Ī: "Ī",
  Ĭ: "Ĭ",
  Ǐ: "Ǐ",
  Î: "Î",
  İ: "İ",
  Ĵ: "Ĵ",
  Ḱ: "Ḱ",
  Ǩ: "Ǩ",
  Ķ: "Ķ",
  Ĺ: "Ĺ",
  Ľ: "Ľ",
  Ļ: "Ļ",
  Ḿ: "Ḿ",
  Ṁ: "Ṁ",
  Ń: "Ń",
  Ǹ: "Ǹ",
  Ñ: "Ñ",
  Ň: "Ň",
  Ṅ: "Ṅ",
  Ņ: "Ņ",
  Ó: "Ó",
  Ò: "Ò",
  Ö: "Ö",
  Ȫ: "Ȫ",
  Õ: "Õ",
  Ṍ: "Ṍ",
  Ṏ: "Ṏ",
  Ȭ: "Ȭ",
  Ō: "Ō",
  Ṓ: "Ṓ",
  Ṑ: "Ṑ",
  Ŏ: "Ŏ",
  Ǒ: "Ǒ",
  Ô: "Ô",
  Ố: "Ố",
  Ồ: "Ồ",
  Ỗ: "Ỗ",
  Ȯ: "Ȯ",
  Ȱ: "Ȱ",
  Ő: "Ő",
  Ṕ: "Ṕ",
  Ṗ: "Ṗ",
  Ŕ: "Ŕ",
  Ř: "Ř",
  Ṙ: "Ṙ",
  Ŗ: "Ŗ",
  Ś: "Ś",
  Ṥ: "Ṥ",
  Š: "Š",
  Ṧ: "Ṧ",
  Ŝ: "Ŝ",
  Ṡ: "Ṡ",
  Ş: "Ş",
  Ť: "Ť",
  Ṫ: "Ṫ",
  Ţ: "Ţ",
  Ú: "Ú",
  Ù: "Ù",
  Ü: "Ü",
  Ǘ: "Ǘ",
  Ǜ: "Ǜ",
  Ǖ: "Ǖ",
  Ǚ: "Ǚ",
  Ũ: "Ũ",
  Ṹ: "Ṹ",
  Ū: "Ū",
  Ṻ: "Ṻ",
  Ŭ: "Ŭ",
  Ǔ: "Ǔ",
  Û: "Û",
  Ů: "Ů",
  Ű: "Ű",
  Ṽ: "Ṽ",
  Ẃ: "Ẃ",
  Ẁ: "Ẁ",
  Ẅ: "Ẅ",
  Ŵ: "Ŵ",
  Ẇ: "Ẇ",
  Ẍ: "Ẍ",
  Ẋ: "Ẋ",
  Ý: "Ý",
  Ỳ: "Ỳ",
  Ÿ: "Ÿ",
  Ỹ: "Ỹ",
  Ȳ: "Ȳ",
  Ŷ: "Ŷ",
  Ẏ: "Ẏ",
  Ź: "Ź",
  Ž: "Ž",
  Ẑ: "Ẑ",
  Ż: "Ż",
  ά: "ά",
  ὰ: "ὰ",
  ᾱ: "ᾱ",
  ᾰ: "ᾰ",
  έ: "έ",
  ὲ: "ὲ",
  ή: "ή",
  ὴ: "ὴ",
  ί: "ί",
  ὶ: "ὶ",
  ϊ: "ϊ",
  ΐ: "ΐ",
  ῒ: "ῒ",
  ῑ: "ῑ",
  ῐ: "ῐ",
  ό: "ό",
  ὸ: "ὸ",
  ύ: "ύ",
  ὺ: "ὺ",
  ϋ: "ϋ",
  ΰ: "ΰ",
  ῢ: "ῢ",
  ῡ: "ῡ",
  ῠ: "ῠ",
  ώ: "ώ",
  ὼ: "ὼ",
  Ύ: "Ύ",
  Ὺ: "Ὺ",
  Ϋ: "Ϋ",
  Ῡ: "Ῡ",
  Ῠ: "Ῠ",
  Ώ: "Ώ",
  Ὼ: "Ὼ"
};
class dn {
  constructor(e, t) {
    this.mode = void 0, this.gullet = void 0, this.settings = void 0, this.leftrightDepth = void 0, this.nextToken = void 0, this.mode = "math", this.gullet = new Nc(e, t, this.mode), this.settings = t, this.leftrightDepth = 0;
  }
  /**
   * Checks a result to make sure it has the right type, and throws an
   * appropriate error otherwise.
   */
  expect(e, t) {
    if (t === void 0 && (t = !0), this.fetch().text !== e)
      throw new L("Expected '" + e + "', got '" + this.fetch().text + "'", this.fetch());
    t && this.consume();
  }
  /**
   * Discards the current lookahead token, considering it consumed.
   */
  consume() {
    this.nextToken = null;
  }
  /**
   * Return the current lookahead token, or if there isn't one (at the
   * beginning, or if the previous lookahead token was consume()d),
   * fetch the next token as the new lookahead token and return it.
   */
  fetch() {
    return this.nextToken == null && (this.nextToken = this.gullet.expandNextToken()), this.nextToken;
  }
  /**
   * Switches between "text" and "math" modes.
   */
  switchMode(e) {
    this.mode = e, this.gullet.switchMode(e);
  }
  /**
   * Main parsing function, which parses an entire input.
   */
  parse() {
    this.settings.globalGroup || this.gullet.beginGroup(), this.settings.colorIsTextColor && this.gullet.macros.set("\\color", "\\textcolor");
    try {
      var e = this.parseExpression(!1);
      return this.expect("EOF"), this.settings.globalGroup || this.gullet.endGroup(), e;
    } finally {
      this.gullet.endGroups();
    }
  }
  /**
   * Fully parse a separate sequence of tokens as a separate job.
   * Tokens should be specified in reverse order, as in a MacroDefinition.
   */
  subparse(e) {
    var t = this.nextToken;
    this.consume(), this.gullet.pushToken(new ot("}")), this.gullet.pushTokens(e);
    var r = this.parseExpression(!1);
    return this.expect("}"), this.nextToken = t, r;
  }
  /**
   * Parses an "expression", which is a list of atoms.
   *
   * `breakOnInfix`: Should the parsing stop when we hit infix nodes? This
   *                 happens when functions have higher precedence han infix
   *                 nodes in implicit parses.
   *
   * `breakOnTokenText`: The text of the token that the expression should end
   *                     with, or `null` if something else should end the
   *                     expression.
   */
  parseExpression(e, t) {
    for (var r = []; ; ) {
      this.mode === "math" && this.consumeSpaces();
      var a = this.fetch();
      if (dn.endOfExpression.indexOf(a.text) !== -1 || t && a.text === t || e && c0[a.text] && c0[a.text].infix)
        break;
      var i = this.parseAtom(t);
      if (i) {
        if (i.type === "internal")
          continue;
      } else break;
      r.push(i);
    }
    return this.mode === "text" && this.formLigatures(r), this.handleInfixNodes(r);
  }
  /**
   * Rewrites infix operators such as \over with corresponding commands such
   * as \frac.
   *
   * There can only be one infix operator per group.  If there's more than one
   * then the expression is ambiguous.  This can be resolved by adding {}.
   */
  handleInfixNodes(e) {
    for (var t = -1, r, a = 0; a < e.length; a++)
      if (e[a].type === "infix") {
        if (t !== -1)
          throw new L("only one infix operator per group", e[a].token);
        t = a, r = e[a].replaceWith;
      }
    if (t !== -1 && r) {
      var i, l, s = e.slice(0, t), u = e.slice(t + 1);
      s.length === 1 && s[0].type === "ordgroup" ? i = s[0] : i = {
        type: "ordgroup",
        mode: this.mode,
        body: s
      }, u.length === 1 && u[0].type === "ordgroup" ? l = u[0] : l = {
        type: "ordgroup",
        mode: this.mode,
        body: u
      };
      var h;
      return r === "\\\\abovefrac" ? h = this.callFunction(r, [i, e[t], l], []) : h = this.callFunction(r, [i, l], []), [h];
    } else
      return e;
  }
  /**
   * Handle a subscript or superscript with nice errors.
   */
  handleSupSubscript(e) {
    var t = this.fetch(), r = t.text;
    this.consume(), this.consumeSpaces();
    var a;
    do {
      var i;
      a = this.parseGroup(e);
    } while (((i = a) == null ? void 0 : i.type) === "internal");
    if (!a)
      throw new L("Expected group after '" + r + "'", t);
    return a;
  }
  /**
   * Converts the textual input of an unsupported command into a text node
   * contained within a color node whose color is determined by errorColor
   */
  formatUnsupportedCmd(e) {
    for (var t = [], r = 0; r < e.length; r++)
      t.push({
        type: "textord",
        mode: "text",
        text: e[r]
      });
    var a = {
      type: "text",
      mode: this.mode,
      body: t
    }, i = {
      type: "color",
      mode: this.mode,
      color: this.settings.errorColor,
      body: [a]
    };
    return i;
  }
  /**
   * Parses a group with optional super/subscripts.
   */
  parseAtom(e) {
    var t = this.parseGroup("atom", e);
    if ((t == null ? void 0 : t.type) === "internal" || this.mode === "text")
      return t;
    for (var r, a; ; ) {
      this.consumeSpaces();
      var i = this.fetch();
      if (i.text === "\\limits" || i.text === "\\nolimits") {
        if (t && t.type === "op") {
          var l = i.text === "\\limits";
          t.limits = l, t.alwaysHandleSupSub = !0;
        } else if (t && t.type === "operatorname")
          t.alwaysHandleSupSub && (t.limits = i.text === "\\limits");
        else
          throw new L("Limit controls must follow a math operator", i);
        this.consume();
      } else if (i.text === "^") {
        if (r)
          throw new L("Double superscript", i);
        r = this.handleSupSubscript("superscript");
      } else if (i.text === "_") {
        if (a)
          throw new L("Double subscript", i);
        a = this.handleSupSubscript("subscript");
      } else if (i.text === "'") {
        if (r)
          throw new L("Double superscript", i);
        var s = {
          type: "textord",
          mode: this.mode,
          text: "\\prime"
        }, u = [s];
        for (this.consume(); this.fetch().text === "'"; )
          u.push(s), this.consume();
        this.fetch().text === "^" && u.push(this.handleSupSubscript("superscript")), r = {
          type: "ordgroup",
          mode: this.mode,
          body: u
        };
      } else if (Er[i.text]) {
        var h = nl.test(i.text), d = [];
        for (d.push(new ot(Er[i.text])), this.consume(); ; ) {
          var g = this.fetch().text;
          if (!Er[g] || nl.test(g) !== h)
            break;
          d.unshift(new ot(Er[g])), this.consume();
        }
        var p = this.subparse(d);
        h ? a = {
          type: "ordgroup",
          mode: "math",
          body: p
        } : r = {
          type: "ordgroup",
          mode: "math",
          body: p
        };
      } else
        break;
    }
    return r || a ? {
      type: "supsub",
      mode: this.mode,
      base: t,
      sup: r,
      sub: a
    } : t;
  }
  /**
   * Parses an entire function, including its base and all of its arguments.
   */
  parseFunction(e, t) {
    var r = this.fetch(), a = r.text, i = c0[a];
    if (!i)
      return null;
    if (this.consume(), t && t !== "atom" && !i.allowedInArgument)
      throw new L("Got function '" + a + "' with no arguments" + (t ? " as " + t : ""), r);
    if (this.mode === "text" && !i.allowedInText)
      throw new L("Can't use function '" + a + "' in text mode", r);
    if (this.mode === "math" && i.allowedInMath === !1)
      throw new L("Can't use function '" + a + "' in math mode", r);
    var {
      args: l,
      optArgs: s
    } = this.parseArguments(a, i);
    return this.callFunction(a, l, s, r, e);
  }
  /**
   * Call a function handler with a suitable context and arguments.
   */
  callFunction(e, t, r, a, i) {
    var l = {
      funcName: e,
      parser: this,
      token: a,
      breakOnTokenText: i
    }, s = c0[e];
    if (s && s.handler)
      return s.handler(l, t, r);
    throw new L("No function handler for " + e);
  }
  /**
   * Parses the arguments of a function or environment
   */
  parseArguments(e, t) {
    var r = t.numArgs + t.numOptionalArgs;
    if (r === 0)
      return {
        args: [],
        optArgs: []
      };
    for (var a = [], i = [], l = 0; l < r; l++) {
      var s = t.argTypes && t.argTypes[l], u = l < t.numOptionalArgs;
      (t.primitive && s == null || // \sqrt expands into primitive if optional argument doesn't exist
      t.type === "sqrt" && l === 1 && i[0] == null) && (s = "primitive");
      var h = this.parseGroupOfType("argument to '" + e + "'", s, u);
      if (u)
        i.push(h);
      else if (h != null)
        a.push(h);
      else
        throw new L("Null argument, please report this as a bug");
    }
    return {
      args: a,
      optArgs: i
    };
  }
  /**
   * Parses a group when the mode is changing.
   */
  parseGroupOfType(e, t, r) {
    switch (t) {
      case "color":
        return this.parseColorGroup(r);
      case "size":
        return this.parseSizeGroup(r);
      case "url":
        return this.parseUrlGroup(r);
      case "math":
      case "text":
        return this.parseArgumentGroup(r, t);
      case "hbox": {
        var a = this.parseArgumentGroup(r, "text");
        return a != null ? {
          type: "styling",
          mode: a.mode,
          body: [a],
          style: "text"
          // simulate \textstyle
        } : null;
      }
      case "raw": {
        var i = this.parseStringGroup("raw", r);
        return i != null ? {
          type: "raw",
          mode: "text",
          string: i.text
        } : null;
      }
      case "primitive": {
        if (r)
          throw new L("A primitive argument cannot be optional");
        var l = this.parseGroup(e);
        if (l == null)
          throw new L("Expected group as " + e, this.fetch());
        return l;
      }
      case "original":
      case null:
      case void 0:
        return this.parseArgumentGroup(r);
      default:
        throw new L("Unknown group type as " + e, this.fetch());
    }
  }
  /**
   * Discard any space tokens, fetching the next non-space token.
   */
  consumeSpaces() {
    for (; this.fetch().text === " "; )
      this.consume();
  }
  /**
   * Parses a group, essentially returning the string formed by the
   * brace-enclosed tokens plus some position information.
   */
  parseStringGroup(e, t) {
    var r = this.gullet.scanArgument(t);
    if (r == null)
      return null;
    for (var a = "", i; (i = this.fetch()).text !== "EOF"; )
      a += i.text, this.consume();
    return this.consume(), r.text = a, r;
  }
  /**
   * Parses a regex-delimited group: the largest sequence of tokens
   * whose concatenated strings match `regex`. Returns the string
   * formed by the tokens plus some position information.
   */
  parseRegexGroup(e, t) {
    for (var r = this.fetch(), a = r, i = "", l; (l = this.fetch()).text !== "EOF" && e.test(i + l.text); )
      a = l, i += a.text, this.consume();
    if (i === "")
      throw new L("Invalid " + t + ": '" + r.text + "'", r);
    return r.range(a, i);
  }
  /**
   * Parses a color description.
   */
  parseColorGroup(e) {
    var t = this.parseStringGroup("color", e);
    if (t == null)
      return null;
    var r = /^(#[a-f0-9]{3}|#?[a-f0-9]{6}|[a-z]+)$/i.exec(t.text);
    if (!r)
      throw new L("Invalid color: '" + t.text + "'", t);
    var a = r[0];
    return /^[0-9a-f]{6}$/i.test(a) && (a = "#" + a), {
      type: "color-token",
      mode: this.mode,
      color: a
    };
  }
  /**
   * Parses a size specification, consisting of magnitude and unit.
   */
  parseSizeGroup(e) {
    var t, r = !1;
    if (this.gullet.consumeSpaces(), !e && this.gullet.future().text !== "{" ? t = this.parseRegexGroup(/^[-+]? *(?:$|\d+|\d+\.\d*|\.\d*) *[a-z]{0,2} *$/, "size") : t = this.parseStringGroup("size", e), !t)
      return null;
    !e && t.text.length === 0 && (t.text = "0pt", r = !0);
    var a = /([-+]?) *(\d+(?:\.\d*)?|\.\d+) *([a-z]{2})/.exec(t.text);
    if (!a)
      throw new L("Invalid size: '" + t.text + "'", t);
    var i = {
      number: +(a[1] + a[2]),
      // sign + magnitude, cast to number
      unit: a[3]
    };
    if (!Es(i))
      throw new L("Invalid unit: '" + i.unit + "'", t);
    return {
      type: "size",
      mode: this.mode,
      value: i,
      isBlank: r
    };
  }
  /**
   * Parses an URL, checking escaped letters and allowed protocols,
   * and setting the catcode of % as an active character (as in \hyperref).
   */
  parseUrlGroup(e) {
    this.gullet.lexer.setCatcode("%", 13), this.gullet.lexer.setCatcode("~", 12);
    var t = this.parseStringGroup("url", e);
    if (this.gullet.lexer.setCatcode("%", 14), this.gullet.lexer.setCatcode("~", 13), t == null)
      return null;
    var r = t.text.replace(/\\([#$%&~_^{}])/g, "$1");
    return {
      type: "url",
      mode: this.mode,
      url: r
    };
  }
  /**
   * Parses an argument with the mode specified.
   */
  parseArgumentGroup(e, t) {
    var r = this.gullet.scanArgument(e);
    if (r == null)
      return null;
    var a = this.mode;
    t && this.switchMode(t), this.gullet.beginGroup();
    var i = this.parseExpression(!1, "EOF");
    this.expect("EOF"), this.gullet.endGroup();
    var l = {
      type: "ordgroup",
      mode: this.mode,
      loc: r.loc,
      body: i
    };
    return t && this.switchMode(a), l;
  }
  /**
   * Parses an ordinary group, which is either a single nucleus (like "x")
   * or an expression in braces (like "{x+y}") or an implicit group, a group
   * that starts at the current position, and ends right before a higher explicit
   * group ends, or at EOF.
   */
  parseGroup(e, t) {
    var r = this.fetch(), a = r.text, i;
    if (a === "{" || a === "\\begingroup") {
      this.consume();
      var l = a === "{" ? "}" : "\\endgroup";
      this.gullet.beginGroup();
      var s = this.parseExpression(!1, l), u = this.fetch();
      this.expect(l), this.gullet.endGroup(), i = {
        type: "ordgroup",
        mode: this.mode,
        loc: Je.range(r, u),
        body: s,
        // A group formed by \begingroup...\endgroup is a semi-simple group
        // which doesn't affect spacing in math mode, i.e., is transparent.
        // https://tex.stackexchange.com/questions/1930/when-should-one-
        // use-begingroup-instead-of-bgroup
        semisimple: a === "\\begingroup" || void 0
      };
    } else if (i = this.parseFunction(t, e) || this.parseSymbol(), i == null && a[0] === "\\" && !_o.hasOwnProperty(a)) {
      if (this.settings.throwOnError)
        throw new L("Undefined control sequence: " + a, r);
      i = this.formatUnsupportedCmd(a), this.consume();
    }
    return i;
  }
  /**
   * Form ligature-like combinations of characters for text mode.
   * This includes inputs like "--", "---", "``" and "''".
   * The result will simply replace multiple textord nodes with a single
   * character in each value by a single textord node having multiple
   * characters in its value.  The representation is still ASCII source.
   * The group will be modified in place.
   */
  formLigatures(e) {
    for (var t = e.length - 1, r = 0; r < t; ++r) {
      var a = e[r], i = a.text;
      i === "-" && e[r + 1].text === "-" && (r + 1 < t && e[r + 2].text === "-" ? (e.splice(r, 3, {
        type: "textord",
        mode: "text",
        loc: Je.range(a, e[r + 2]),
        text: "---"
      }), t -= 2) : (e.splice(r, 2, {
        type: "textord",
        mode: "text",
        loc: Je.range(a, e[r + 1]),
        text: "--"
      }), t -= 1)), (i === "'" || i === "`") && e[r + 1].text === i && (e.splice(r, 2, {
        type: "textord",
        mode: "text",
        loc: Je.range(a, e[r + 1]),
        text: i + i
      }), t -= 1);
    }
  }
  /**
   * Parse a single symbol out of the string. Here, we handle single character
   * symbols and special functions like \verb.
   */
  parseSymbol() {
    var e = this.fetch(), t = e.text;
    if (/^\\verb[^a-zA-Z]/.test(t)) {
      this.consume();
      var r = t.slice(5), a = r.charAt(0) === "*";
      if (a && (r = r.slice(1)), r.length < 2 || r.charAt(0) !== r.slice(-1))
        throw new L(`\\verb assertion failed --
                    please report what input caused this bug`);
      return r = r.slice(1, -1), {
        type: "verb",
        mode: "text",
        body: r,
        star: a
      };
    }
    al.hasOwnProperty(t[0]) && !ge[this.mode][t[0]] && (this.settings.strict && this.mode === "math" && this.settings.reportNonstrict("unicodeTextInMathMode", 'Accented Unicode text character "' + t[0] + '" used in math mode', e), t = al[t[0]] + t.slice(1));
    var i = Mc.exec(t);
    i && (t = t.substring(0, i.index), t === "i" ? t = "ı" : t === "j" && (t = "ȷ"));
    var l;
    if (ge[this.mode][t]) {
      this.settings.strict && this.mode === "math" && na.indexOf(t) >= 0 && this.settings.reportNonstrict("unicodeTextInMathMode", 'Latin-1/Unicode text character "' + t[0] + '" used in math mode', e);
      var s = ge[this.mode][t].group, u = Je.range(e), h;
      if (D1.hasOwnProperty(s)) {
        var d = s;
        h = {
          type: "atom",
          mode: this.mode,
          family: d,
          loc: u,
          text: t
        };
      } else
        h = {
          type: s,
          mode: this.mode,
          loc: u,
          text: t
        };
      l = h;
    } else if (t.charCodeAt(0) >= 128)
      this.settings.strict && (Ss(t.charCodeAt(0)) ? this.mode === "math" && this.settings.reportNonstrict("unicodeTextInMathMode", 'Unicode text character "' + t[0] + '" used in math mode', e) : this.settings.reportNonstrict("unknownSymbol", 'Unrecognized Unicode character "' + t[0] + '"' + (" (" + t.charCodeAt(0) + ")"), e)), l = {
        type: "textord",
        mode: "text",
        loc: Je.range(e),
        text: t
      };
    else
      return null;
    if (this.consume(), i)
      for (var g = 0; g < i[0].length; g++) {
        var p = i[0][g];
        if (!Ln[p])
          throw new L("Unknown accent ' " + p + "'", e);
        var v = Ln[p][this.mode] || Ln[p].text;
        if (!v)
          throw new L("Accent " + p + " unsupported in " + this.mode + " mode", e);
        l = {
          type: "accent",
          mode: this.mode,
          loc: Je.range(e),
          label: v,
          isStretchy: !1,
          isShifty: !0,
          // $FlowFixMe
          base: l
        };
      }
    return l;
  }
}
dn.endOfExpression = ["}", "\\endgroup", "\\end", "\\right", "&"];
var Pa = function(e, t) {
  if (!(typeof e == "string" || e instanceof String))
    throw new TypeError("KaTeX can only parse string typed expression");
  var r = new dn(e, t);
  delete r.gullet.macros.current["\\df@tag"];
  var a = r.parse();
  if (delete r.gullet.macros.current["\\current@color"], delete r.gullet.macros.current["\\color"], r.gullet.macros.get("\\df@tag")) {
    if (!t.displayMode)
      throw new L("\\tag works only in display equations");
    a = [{
      type: "tag",
      mode: "text",
      body: a,
      tag: r.subparse([new ot("\\df@tag")])
    }];
  }
  return a;
}, Ha = function(e, t, r) {
  t.textContent = "";
  var a = mn(e, r).toNode();
  t.appendChild(a);
};
typeof document < "u" && document.compatMode !== "CSS1Compat" && (typeof console < "u" && console.warn("Warning: KaTeX doesn't work in quirks mode. Make sure your website has a suitable doctype."), Ha = function() {
  throw new L("KaTeX doesn't work in quirks mode.");
});
var bo = function(e, t) {
  var r = mn(e, t).toMarkup();
  return r;
}, yo = function(e, t) {
  var r = new ka(t);
  return Pa(e, r);
}, wo = function(e, t, r) {
  if (r.throwOnError || !(e instanceof L))
    throw e;
  var a = F.makeSpan(["katex-error"], [new ut(t)]);
  return a.setAttribute("title", e.toString()), a.setAttribute("style", "color:" + r.errorColor), a;
}, mn = function(e, t) {
  var r = new ka(t);
  try {
    var a = Pa(e, r);
    return j1(a, e, r);
  } catch (i) {
    return wo(i, e, r);
  }
}, xo = function(e, t) {
  var r = new ka(t);
  try {
    var a = Pa(e, r);
    return Y1(a, e, r);
  } catch (i) {
    return wo(i, e, r);
  }
}, ko = "0.16.22", Do = {
  Span: cr,
  Anchor: Aa,
  SymbolNode: ut,
  SvgNode: r0,
  PathNode: m0,
  LineNode: ra
}, ua = {
  /**
   * Current KaTeX version
   */
  version: ko,
  /**
   * Renders the given LaTeX into an HTML+MathML combination, and adds
   * it as a child to the specified DOM node.
   */
  render: Ha,
  /**
   * Renders the given LaTeX into an HTML+MathML combination string,
   * for sending to the client.
   */
  renderToString: bo,
  /**
   * KaTeX error, usually during parsing.
   */
  ParseError: L,
  /**
   * The schema of Settings
   */
  SETTINGS_SCHEMA: er,
  /**
   * Parses the given LaTeX into KaTeX's internal parse tree structure,
   * without rendering to HTML or MathML.
   *
   * NOTE: This method is not currently recommended for public use.
   * The internal tree representation is unstable and is very likely
   * to change. Use at your own risk.
   */
  __parse: yo,
  /**
   * Renders the given LaTeX into an HTML+MathML internal DOM tree
   * representation, without flattening that representation to a string.
   *
   * NOTE: This method is not currently recommended for public use.
   * The internal tree representation is unstable and is very likely
   * to change. Use at your own risk.
   */
  __renderToDomTree: mn,
  /**
   * Renders the given LaTeX into an HTML internal DOM tree representation,
   * without MathML and without flattening that representation to a string.
   *
   * NOTE: This method is not currently recommended for public use.
   * The internal tree representation is unstable and is very likely
   * to change. Use at your own risk.
   */
  __renderToHTMLTree: xo,
  /**
   * extends internal font metrics object with a new object
   * each key in the new object represents a font name
  */
  __setFontMetrics: As,
  /**
   * adds a new symbol to builtin symbols table
   */
  __defineSymbol: o,
  /**
   * adds a new function to builtin function list,
   * which directly produce parse tree elements
   * and have their own html/mathml builders
   */
  __defineFunction: H,
  /**
   * adds a new macro to builtin macro list
   */
  __defineMacro: f,
  /**
   * Expose the dom tree node types, which can be useful for type checking nodes.
   *
   * NOTE: These methods are not currently recommended for public use.
   * The internal tree representation is unstable and is very likely
   * to change. Use at your own risk.
   */
  __domTree: Do
};
const m2 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  ParseError: L,
  SETTINGS_SCHEMA: er,
  __defineFunction: H,
  __defineMacro: f,
  __defineSymbol: o,
  __domTree: Do,
  __parse: yo,
  __renderToDomTree: mn,
  __renderToHTMLTree: xo,
  __setFontMetrics: As,
  default: ua,
  get render() {
    return Ha;
  },
  renderToString: bo,
  version: ko
}, Symbol.toStringTag, { value: "Module" }));
var qc = function(e, t, r) {
  for (var a = r, i = 0, l = e.length; a < t.length; ) {
    var s = t[a];
    if (i <= 0 && t.slice(a, a + l) === e)
      return a;
    s === "\\" ? a++ : s === "{" ? i++ : s === "}" && i--, a++;
  }
  return -1;
}, Lc = function(e) {
  return e.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&");
}, Ic = /^\\begin{/, Oc = function(e, t) {
  for (var r, a = [], i = new RegExp("(" + t.map((h) => Lc(h.left)).join("|") + ")"); r = e.search(i), r !== -1; ) {
    r > 0 && (a.push({
      type: "text",
      data: e.slice(0, r)
    }), e = e.slice(r));
    var l = t.findIndex((h) => e.startsWith(h.left));
    if (r = qc(t[l].right, e, t[l].left.length), r === -1)
      break;
    var s = e.slice(0, r + t[l].right.length), u = Ic.test(s) ? s : e.slice(t[l].left.length, r);
    a.push({
      type: "math",
      data: u,
      rawData: s,
      display: t[l].display
    }), e = e.slice(r + t[l].right.length);
  }
  return e !== "" && a.push({
    type: "text",
    data: e
  }), a;
}, Pc = function(e, t) {
  var r = Oc(e, t.delimiters);
  if (r.length === 1 && r[0].type === "text")
    return null;
  for (var a = document.createDocumentFragment(), i = 0; i < r.length; i++)
    if (r[i].type === "text")
      a.appendChild(document.createTextNode(r[i].data));
    else {
      var l = document.createElement("span"), s = r[i].data;
      t.displayMode = r[i].display;
      try {
        t.preProcess && (s = t.preProcess(s)), ua.render(s, l, t);
      } catch (u) {
        if (!(u instanceof ua.ParseError))
          throw u;
        t.errorCallback("KaTeX auto-render: Failed to parse `" + r[i].data + "` with ", u), a.appendChild(document.createTextNode(r[i].rawData));
        continue;
      }
      a.appendChild(l);
    }
  return a;
}, Hc = function n(e, t) {
  for (var r = 0; r < e.childNodes.length; r++) {
    var a = e.childNodes[r];
    if (a.nodeType === 3) {
      for (var i = a.textContent, l = a.nextSibling, s = 0; l && l.nodeType === Node.TEXT_NODE; )
        i += l.textContent, l = l.nextSibling, s++;
      var u = Pc(i, t);
      if (u) {
        for (var h = 0; h < s; h++)
          a.nextSibling.remove();
        r += u.childNodes.length - 1, e.replaceChild(u, a);
      } else
        r += s;
    } else a.nodeType === 1 && function() {
      var d = " " + a.className + " ", g = t.ignoredTags.indexOf(a.nodeName.toLowerCase()) === -1 && t.ignoredClasses.every((p) => d.indexOf(" " + p + " ") === -1);
      g && n(a, t);
    }();
  }
}, Uc = function(e, t) {
  if (!e)
    throw new Error("No element provided to render");
  var r = {};
  for (var a in t)
    t.hasOwnProperty(a) && (r[a] = t[a]);
  r.delimiters = r.delimiters || [
    {
      left: "$$",
      right: "$$",
      display: !0
    },
    {
      left: "\\(",
      right: "\\)",
      display: !1
    },
    // LaTeX uses $…$, but it ruins the display of normal `$` in text:
    // {left: "$", right: "$", display: false},
    // $ must come after $$
    // Render AMS environments even if outside $$…$$ delimiters.
    {
      left: "\\begin{equation}",
      right: "\\end{equation}",
      display: !0
    },
    {
      left: "\\begin{align}",
      right: "\\end{align}",
      display: !0
    },
    {
      left: "\\begin{alignat}",
      right: "\\end{alignat}",
      display: !0
    },
    {
      left: "\\begin{gather}",
      right: "\\end{gather}",
      display: !0
    },
    {
      left: "\\begin{CD}",
      right: "\\end{CD}",
      display: !0
    },
    {
      left: "\\[",
      right: "\\]",
      display: !0
    }
  ], r.ignoredTags = r.ignoredTags || ["script", "noscript", "style", "textarea", "pre", "code", "option"], r.ignoredClasses = r.ignoredClasses || [], r.errorCallback = r.errorCallback || console.error, r.macros = r.macros || {}, Hc(e, r);
};
function Ua() {
  return {
    async: !1,
    breaks: !1,
    extensions: null,
    gfm: !0,
    hooks: null,
    pedantic: !1,
    renderer: null,
    silent: !1,
    tokenizer: null,
    walkTokens: null
  };
}
let k0 = Ua();
function So(n) {
  k0 = n;
}
const Ao = /[&<>"']/, Gc = new RegExp(Ao.source, "g"), Eo = /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/, Vc = new RegExp(Eo.source, "g"), Wc = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
}, il = (n) => Wc[n];
function et(n, e) {
  if (e) {
    if (Ao.test(n))
      return n.replace(Gc, il);
  } else if (Eo.test(n))
    return n.replace(Vc, il);
  return n;
}
const jc = /&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/ig;
function Yc(n) {
  return n.replace(jc, (e, t) => (t = t.toLowerCase(), t === "colon" ? ":" : t.charAt(0) === "#" ? t.charAt(1) === "x" ? String.fromCharCode(parseInt(t.substring(2), 16)) : String.fromCharCode(+t.substring(1)) : ""));
}
const Xc = /(^|[^\[])\^/g;
function me(n, e) {
  let t = typeof n == "string" ? n : n.source;
  e = e || "";
  const r = {
    replace: (a, i) => {
      let l = typeof i == "string" ? i : i.source;
      return l = l.replace(Xc, "$1"), t = t.replace(a, l), r;
    },
    getRegex: () => new RegExp(t, e)
  };
  return r;
}
function ll(n) {
  try {
    n = encodeURI(n).replace(/%25/g, "%");
  } catch {
    return null;
  }
  return n;
}
const rr = { exec: () => null };
function sl(n, e) {
  const t = n.replace(/\|/g, (i, l, s) => {
    let u = !1, h = l;
    for (; --h >= 0 && s[h] === "\\"; )
      u = !u;
    return u ? "|" : " |";
  }), r = t.split(/ \|/);
  let a = 0;
  if (r[0].trim() || r.shift(), r.length > 0 && !r[r.length - 1].trim() && r.pop(), e)
    if (r.length > e)
      r.splice(e);
    else
      for (; r.length < e; )
        r.push("");
  for (; a < r.length; a++)
    r[a] = r[a].trim().replace(/\\\|/g, "|");
  return r;
}
function Fr(n, e, t) {
  const r = n.length;
  if (r === 0)
    return "";
  let a = 0;
  for (; a < r && n.charAt(r - a - 1) === e; )
    a++;
  return n.slice(0, r - a);
}
function Zc(n, e) {
  if (n.indexOf(e[1]) === -1)
    return -1;
  let t = 0;
  for (let r = 0; r < n.length; r++)
    if (n[r] === "\\")
      r++;
    else if (n[r] === e[0])
      t++;
    else if (n[r] === e[1] && (t--, t < 0))
      return r;
  return -1;
}
function ol(n, e, t, r) {
  const a = e.href, i = e.title ? et(e.title) : null, l = n[1].replace(/\\([\[\]])/g, "$1");
  if (n[0].charAt(0) !== "!") {
    r.state.inLink = !0;
    const s = {
      type: "link",
      raw: t,
      href: a,
      title: i,
      text: l,
      tokens: r.inlineTokens(l)
    };
    return r.state.inLink = !1, s;
  }
  return {
    type: "image",
    raw: t,
    href: a,
    title: i,
    text: et(l)
  };
}
function Kc(n, e) {
  const t = n.match(/^(\s+)(?:```)/);
  if (t === null)
    return e;
  const r = t[1];
  return e.split(`
`).map((a) => {
    const i = a.match(/^\s+/);
    if (i === null)
      return a;
    const [l] = i;
    return l.length >= r.length ? a.slice(r.length) : a;
  }).join(`
`);
}
class tn {
  // set by the lexer
  constructor(e) {
    _e(this, "options");
    _e(this, "rules");
    // set by the lexer
    _e(this, "lexer");
    this.options = e || k0;
  }
  space(e) {
    const t = this.rules.block.newline.exec(e);
    if (t && t[0].length > 0)
      return {
        type: "space",
        raw: t[0]
      };
  }
  code(e) {
    const t = this.rules.block.code.exec(e);
    if (t) {
      const r = t[0].replace(/^ {1,4}/gm, "");
      return {
        type: "code",
        raw: t[0],
        codeBlockStyle: "indented",
        text: this.options.pedantic ? r : Fr(r, `
`)
      };
    }
  }
  fences(e) {
    const t = this.rules.block.fences.exec(e);
    if (t) {
      const r = t[0], a = Kc(r, t[3] || "");
      return {
        type: "code",
        raw: r,
        lang: t[2] ? t[2].trim().replace(this.rules.inline.anyPunctuation, "$1") : t[2],
        text: a
      };
    }
  }
  heading(e) {
    const t = this.rules.block.heading.exec(e);
    if (t) {
      let r = t[2].trim();
      if (/#$/.test(r)) {
        const a = Fr(r, "#");
        (this.options.pedantic || !a || / $/.test(a)) && (r = a.trim());
      }
      return {
        type: "heading",
        raw: t[0],
        depth: t[1].length,
        text: r,
        tokens: this.lexer.inline(r)
      };
    }
  }
  hr(e) {
    const t = this.rules.block.hr.exec(e);
    if (t)
      return {
        type: "hr",
        raw: t[0]
      };
  }
  blockquote(e) {
    const t = this.rules.block.blockquote.exec(e);
    if (t) {
      let r = t[0].replace(/\n {0,3}((?:=+|-+) *)(?=\n|$)/g, `
    $1`);
      r = Fr(r.replace(/^ *>[ \t]?/gm, ""), `
`);
      const a = this.lexer.state.top;
      this.lexer.state.top = !0;
      const i = this.lexer.blockTokens(r);
      return this.lexer.state.top = a, {
        type: "blockquote",
        raw: t[0],
        tokens: i,
        text: r
      };
    }
  }
  list(e) {
    let t = this.rules.block.list.exec(e);
    if (t) {
      let r = t[1].trim();
      const a = r.length > 1, i = {
        type: "list",
        raw: "",
        ordered: a,
        start: a ? +r.slice(0, -1) : "",
        loose: !1,
        items: []
      };
      r = a ? `\\d{1,9}\\${r.slice(-1)}` : `\\${r}`, this.options.pedantic && (r = a ? r : "[*+-]");
      const l = new RegExp(`^( {0,3}${r})((?:[	 ][^\\n]*)?(?:\\n|$))`);
      let s = "", u = "", h = !1;
      for (; e; ) {
        let d = !1;
        if (!(t = l.exec(e)) || this.rules.block.hr.test(e))
          break;
        s = t[0], e = e.substring(s.length);
        let g = t[2].split(`
`, 1)[0].replace(/^\t+/, (z) => " ".repeat(3 * z.length)), p = e.split(`
`, 1)[0], v = 0;
        this.options.pedantic ? (v = 2, u = g.trimStart()) : (v = t[2].search(/[^ ]/), v = v > 4 ? 1 : v, u = g.slice(v), v += t[1].length);
        let k = !1;
        if (!g && /^ *$/.test(p) && (s += p + `
`, e = e.substring(p.length + 1), d = !0), !d) {
          const z = new RegExp(`^ {0,${Math.min(3, v - 1)}}(?:[*+-]|\\d{1,9}[.)])((?:[ 	][^\\n]*)?(?:\\n|$))`), x = new RegExp(`^ {0,${Math.min(3, v - 1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`), _ = new RegExp(`^ {0,${Math.min(3, v - 1)}}(?:\`\`\`|~~~)`), w = new RegExp(`^ {0,${Math.min(3, v - 1)}}#`);
          for (; e; ) {
            const E = e.split(`
`, 1)[0];
            if (p = E, this.options.pedantic && (p = p.replace(/^ {1,4}(?=( {4})*[^ ])/g, "  ")), _.test(p) || w.test(p) || z.test(p) || x.test(e))
              break;
            if (p.search(/[^ ]/) >= v || !p.trim())
              u += `
` + p.slice(v);
            else {
              if (k || g.search(/[^ ]/) >= 4 || _.test(g) || w.test(g) || x.test(g))
                break;
              u += `
` + p;
            }
            !k && !p.trim() && (k = !0), s += E + `
`, e = e.substring(E.length + 1), g = p.slice(v);
          }
        }
        i.loose || (h ? i.loose = !0 : /\n *\n *$/.test(s) && (h = !0));
        let A = null, C;
        this.options.gfm && (A = /^\[[ xX]\] /.exec(u), A && (C = A[0] !== "[ ] ", u = u.replace(/^\[[ xX]\] +/, ""))), i.items.push({
          type: "list_item",
          raw: s,
          task: !!A,
          checked: C,
          loose: !1,
          text: u,
          tokens: []
        }), i.raw += s;
      }
      i.items[i.items.length - 1].raw = s.trimEnd(), i.items[i.items.length - 1].text = u.trimEnd(), i.raw = i.raw.trimEnd();
      for (let d = 0; d < i.items.length; d++)
        if (this.lexer.state.top = !1, i.items[d].tokens = this.lexer.blockTokens(i.items[d].text, []), !i.loose) {
          const g = i.items[d].tokens.filter((v) => v.type === "space"), p = g.length > 0 && g.some((v) => /\n.*\n/.test(v.raw));
          i.loose = p;
        }
      if (i.loose)
        for (let d = 0; d < i.items.length; d++)
          i.items[d].loose = !0;
      return i;
    }
  }
  html(e) {
    const t = this.rules.block.html.exec(e);
    if (t)
      return {
        type: "html",
        block: !0,
        raw: t[0],
        pre: t[1] === "pre" || t[1] === "script" || t[1] === "style",
        text: t[0]
      };
  }
  def(e) {
    const t = this.rules.block.def.exec(e);
    if (t) {
      const r = t[1].toLowerCase().replace(/\s+/g, " "), a = t[2] ? t[2].replace(/^<(.*)>$/, "$1").replace(this.rules.inline.anyPunctuation, "$1") : "", i = t[3] ? t[3].substring(1, t[3].length - 1).replace(this.rules.inline.anyPunctuation, "$1") : t[3];
      return {
        type: "def",
        tag: r,
        raw: t[0],
        href: a,
        title: i
      };
    }
  }
  table(e) {
    const t = this.rules.block.table.exec(e);
    if (!t || !/[:|]/.test(t[2]))
      return;
    const r = sl(t[1]), a = t[2].replace(/^\||\| *$/g, "").split("|"), i = t[3] && t[3].trim() ? t[3].replace(/\n[ \t]*$/, "").split(`
`) : [], l = {
      type: "table",
      raw: t[0],
      header: [],
      align: [],
      rows: []
    };
    if (r.length === a.length) {
      for (const s of a)
        /^ *-+: *$/.test(s) ? l.align.push("right") : /^ *:-+: *$/.test(s) ? l.align.push("center") : /^ *:-+ *$/.test(s) ? l.align.push("left") : l.align.push(null);
      for (const s of r)
        l.header.push({
          text: s,
          tokens: this.lexer.inline(s)
        });
      for (const s of i)
        l.rows.push(sl(s, l.header.length).map((u) => ({
          text: u,
          tokens: this.lexer.inline(u)
        })));
      return l;
    }
  }
  lheading(e) {
    const t = this.rules.block.lheading.exec(e);
    if (t)
      return {
        type: "heading",
        raw: t[0],
        depth: t[2].charAt(0) === "=" ? 1 : 2,
        text: t[1],
        tokens: this.lexer.inline(t[1])
      };
  }
  paragraph(e) {
    const t = this.rules.block.paragraph.exec(e);
    if (t) {
      const r = t[1].charAt(t[1].length - 1) === `
` ? t[1].slice(0, -1) : t[1];
      return {
        type: "paragraph",
        raw: t[0],
        text: r,
        tokens: this.lexer.inline(r)
      };
    }
  }
  text(e) {
    const t = this.rules.block.text.exec(e);
    if (t)
      return {
        type: "text",
        raw: t[0],
        text: t[0],
        tokens: this.lexer.inline(t[0])
      };
  }
  escape(e) {
    const t = this.rules.inline.escape.exec(e);
    if (t)
      return {
        type: "escape",
        raw: t[0],
        text: et(t[1])
      };
  }
  tag(e) {
    const t = this.rules.inline.tag.exec(e);
    if (t)
      return !this.lexer.state.inLink && /^<a /i.test(t[0]) ? this.lexer.state.inLink = !0 : this.lexer.state.inLink && /^<\/a>/i.test(t[0]) && (this.lexer.state.inLink = !1), !this.lexer.state.inRawBlock && /^<(pre|code|kbd|script)(\s|>)/i.test(t[0]) ? this.lexer.state.inRawBlock = !0 : this.lexer.state.inRawBlock && /^<\/(pre|code|kbd|script)(\s|>)/i.test(t[0]) && (this.lexer.state.inRawBlock = !1), {
        type: "html",
        raw: t[0],
        inLink: this.lexer.state.inLink,
        inRawBlock: this.lexer.state.inRawBlock,
        block: !1,
        text: t[0]
      };
  }
  link(e) {
    const t = this.rules.inline.link.exec(e);
    if (t) {
      const r = t[2].trim();
      if (!this.options.pedantic && /^</.test(r)) {
        if (!/>$/.test(r))
          return;
        const l = Fr(r.slice(0, -1), "\\");
        if ((r.length - l.length) % 2 === 0)
          return;
      } else {
        const l = Zc(t[2], "()");
        if (l > -1) {
          const u = (t[0].indexOf("!") === 0 ? 5 : 4) + t[1].length + l;
          t[2] = t[2].substring(0, l), t[0] = t[0].substring(0, u).trim(), t[3] = "";
        }
      }
      let a = t[2], i = "";
      if (this.options.pedantic) {
        const l = /^([^'"]*[^\s])\s+(['"])(.*)\2/.exec(a);
        l && (a = l[1], i = l[3]);
      } else
        i = t[3] ? t[3].slice(1, -1) : "";
      return a = a.trim(), /^</.test(a) && (this.options.pedantic && !/>$/.test(r) ? a = a.slice(1) : a = a.slice(1, -1)), ol(t, {
        href: a && a.replace(this.rules.inline.anyPunctuation, "$1"),
        title: i && i.replace(this.rules.inline.anyPunctuation, "$1")
      }, t[0], this.lexer);
    }
  }
  reflink(e, t) {
    let r;
    if ((r = this.rules.inline.reflink.exec(e)) || (r = this.rules.inline.nolink.exec(e))) {
      const a = (r[2] || r[1]).replace(/\s+/g, " "), i = t[a.toLowerCase()];
      if (!i) {
        const l = r[0].charAt(0);
        return {
          type: "text",
          raw: l,
          text: l
        };
      }
      return ol(r, i, r[0], this.lexer);
    }
  }
  emStrong(e, t, r = "") {
    let a = this.rules.inline.emStrongLDelim.exec(e);
    if (!a || a[3] && r.match(/[\p{L}\p{N}]/u))
      return;
    if (!(a[1] || a[2] || "") || !r || this.rules.inline.punctuation.exec(r)) {
      const l = [...a[0]].length - 1;
      let s, u, h = l, d = 0;
      const g = a[0][0] === "*" ? this.rules.inline.emStrongRDelimAst : this.rules.inline.emStrongRDelimUnd;
      for (g.lastIndex = 0, t = t.slice(-1 * e.length + l); (a = g.exec(t)) != null; ) {
        if (s = a[1] || a[2] || a[3] || a[4] || a[5] || a[6], !s)
          continue;
        if (u = [...s].length, a[3] || a[4]) {
          h += u;
          continue;
        } else if ((a[5] || a[6]) && l % 3 && !((l + u) % 3)) {
          d += u;
          continue;
        }
        if (h -= u, h > 0)
          continue;
        u = Math.min(u, u + h + d);
        const p = [...a[0]][0].length, v = e.slice(0, l + a.index + p + u);
        if (Math.min(l, u) % 2) {
          const A = v.slice(1, -1);
          return {
            type: "em",
            raw: v,
            text: A,
            tokens: this.lexer.inlineTokens(A)
          };
        }
        const k = v.slice(2, -2);
        return {
          type: "strong",
          raw: v,
          text: k,
          tokens: this.lexer.inlineTokens(k)
        };
      }
    }
  }
  codespan(e) {
    const t = this.rules.inline.code.exec(e);
    if (t) {
      let r = t[2].replace(/\n/g, " ");
      const a = /[^ ]/.test(r), i = /^ /.test(r) && / $/.test(r);
      return a && i && (r = r.substring(1, r.length - 1)), r = et(r, !0), {
        type: "codespan",
        raw: t[0],
        text: r
      };
    }
  }
  br(e) {
    const t = this.rules.inline.br.exec(e);
    if (t)
      return {
        type: "br",
        raw: t[0]
      };
  }
  del(e) {
    const t = this.rules.inline.del.exec(e);
    if (t)
      return {
        type: "del",
        raw: t[0],
        text: t[2],
        tokens: this.lexer.inlineTokens(t[2])
      };
  }
  autolink(e) {
    const t = this.rules.inline.autolink.exec(e);
    if (t) {
      let r, a;
      return t[2] === "@" ? (r = et(t[1]), a = "mailto:" + r) : (r = et(t[1]), a = r), {
        type: "link",
        raw: t[0],
        text: r,
        href: a,
        tokens: [
          {
            type: "text",
            raw: r,
            text: r
          }
        ]
      };
    }
  }
  url(e) {
    var r;
    let t;
    if (t = this.rules.inline.url.exec(e)) {
      let a, i;
      if (t[2] === "@")
        a = et(t[0]), i = "mailto:" + a;
      else {
        let l;
        do
          l = t[0], t[0] = ((r = this.rules.inline._backpedal.exec(t[0])) == null ? void 0 : r[0]) ?? "";
        while (l !== t[0]);
        a = et(t[0]), t[1] === "www." ? i = "http://" + t[0] : i = t[0];
      }
      return {
        type: "link",
        raw: t[0],
        text: a,
        href: i,
        tokens: [
          {
            type: "text",
            raw: a,
            text: a
          }
        ]
      };
    }
  }
  inlineText(e) {
    const t = this.rules.inline.text.exec(e);
    if (t) {
      let r;
      return this.lexer.state.inRawBlock ? r = t[0] : r = et(t[0]), {
        type: "text",
        raw: t[0],
        text: r
      };
    }
  }
}
const Qc = /^(?: *(?:\n|$))+/, Jc = /^( {4}[^\n]+(?:\n(?: *(?:\n|$))*)?)+/, eh = /^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/, dr = /^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/, th = /^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/, Fo = /(?:[*+-]|\d{1,9}[.)])/, Co = me(/^(?!bull |blockCode|fences|blockquote|heading|html)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html))+?)\n {0,3}(=+|-+) *(?:\n+|$)/).replace(/bull/g, Fo).replace(/blockCode/g, / {4}/).replace(/fences/g, / {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g, / {0,3}>/).replace(/heading/g, / {0,3}#{1,6}/).replace(/html/g, / {0,3}<[^\n>]+>\n/).getRegex(), Ga = /^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/, rh = /^[^\n]+/, Va = /(?!\s*\])(?:\\.|[^\[\]\\])+/, nh = me(/^ {0,3}\[(label)\]: *(?:\n *)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n *)?| *\n *)(title))? *(?:\n+|$)/).replace("label", Va).replace("title", /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex(), ah = me(/^( {0,3}bull)([ \t][^\n]+?)?(?:\n|$)/).replace(/bull/g, Fo).getRegex(), fn = "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul", Wa = /<!--(?:-?>|[\s\S]*?(?:-->|$))/, ih = me("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n *)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$))", "i").replace("comment", Wa).replace("tag", fn).replace("attribute", / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex(), To = me(Ga).replace("hr", dr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("|table", "").replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", fn).getRegex(), lh = me(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph", To).getRegex(), ja = {
  blockquote: lh,
  code: Jc,
  def: nh,
  fences: eh,
  heading: th,
  hr: dr,
  html: ih,
  lheading: Co,
  list: ah,
  newline: Qc,
  paragraph: To,
  table: rr,
  text: rh
}, ul = me("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr", dr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("blockquote", " {0,3}>").replace("code", " {4}[^\\n]").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", fn).getRegex(), sh = {
  ...ja,
  table: ul,
  paragraph: me(Ga).replace("hr", dr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("table", ul).replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", fn).getRegex()
}, oh = {
  ...ja,
  html: me(`^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:"[^"]*"|'[^']*'|\\s[^'"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))`).replace("comment", Wa).replace(/tag/g, "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(),
  def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,
  heading: /^(#{1,6})(.*)(?:\n+|$)/,
  fences: rr,
  // fences not supported
  lheading: /^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/,
  paragraph: me(Ga).replace("hr", dr).replace("heading", ` *#{1,6} *[^
]`).replace("lheading", Co).replace("|table", "").replace("blockquote", " {0,3}>").replace("|fences", "").replace("|list", "").replace("|html", "").replace("|tag", "").getRegex()
}, $o = /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/, uh = /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/, Mo = /^( {2,}|\\)\n(?!\s*$)/, ch = /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/, mr = "\\p{P}\\p{S}", hh = me(/^((?![*_])[\spunctuation])/, "u").replace(/punctuation/g, mr).getRegex(), dh = /\[[^[\]]*?\]\([^\(\)]*?\)|`[^`]*?`|<[^<>]*?>/g, mh = me(/^(?:\*+(?:((?!\*)[punct])|[^\s*]))|^_+(?:((?!_)[punct])|([^\s_]))/, "u").replace(/punct/g, mr).getRegex(), fh = me("^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)[punct](\\*+)(?=[\\s]|$)|[^punct\\s](\\*+)(?!\\*)(?=[punct\\s]|$)|(?!\\*)[punct\\s](\\*+)(?=[^punct\\s])|[\\s](\\*+)(?!\\*)(?=[punct])|(?!\\*)[punct](\\*+)(?!\\*)(?=[punct])|[^punct\\s](\\*+)(?=[^punct\\s])", "gu").replace(/punct/g, mr).getRegex(), ph = me("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)[punct](_+)(?=[\\s]|$)|[^punct\\s](_+)(?!_)(?=[punct\\s]|$)|(?!_)[punct\\s](_+)(?=[^punct\\s])|[\\s](_+)(?!_)(?=[punct])|(?!_)[punct](_+)(?!_)(?=[punct])", "gu").replace(/punct/g, mr).getRegex(), gh = me(/\\([punct])/, "gu").replace(/punct/g, mr).getRegex(), vh = me(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme", /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email", /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex(), _h = me(Wa).replace("(?:-->|$)", "-->").getRegex(), bh = me("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment", _h).replace("attribute", /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex(), rn = /(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/, yh = me(/^!?\[(label)\]\(\s*(href)(?:\s+(title))?\s*\)/).replace("label", rn).replace("href", /<(?:\\.|[^\n<>\\])+>|[^\s\x00-\x1f]*/).replace("title", /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex(), zo = me(/^!?\[(label)\]\[(ref)\]/).replace("label", rn).replace("ref", Va).getRegex(), Bo = me(/^!?\[(ref)\](?:\[\])?/).replace("ref", Va).getRegex(), wh = me("reflink|nolink(?!\\()", "g").replace("reflink", zo).replace("nolink", Bo).getRegex(), Ya = {
  _backpedal: rr,
  // only used for GFM url
  anyPunctuation: gh,
  autolink: vh,
  blockSkip: dh,
  br: Mo,
  code: uh,
  del: rr,
  emStrongLDelim: mh,
  emStrongRDelimAst: fh,
  emStrongRDelimUnd: ph,
  escape: $o,
  link: yh,
  nolink: Bo,
  punctuation: hh,
  reflink: zo,
  reflinkSearch: wh,
  tag: bh,
  text: ch,
  url: rr
}, xh = {
  ...Ya,
  link: me(/^!?\[(label)\]\((.*?)\)/).replace("label", rn).getRegex(),
  reflink: me(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label", rn).getRegex()
}, ca = {
  ...Ya,
  escape: me($o).replace("])", "~|])").getRegex(),
  url: me(/^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/, "i").replace("email", /[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(),
  _backpedal: /(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/,
  del: /^(~~?)(?=[^\s~])([\s\S]*?[^\s~])\1(?=[^~]|$)/,
  text: /^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/
}, kh = {
  ...ca,
  br: me(Mo).replace("{2,}", "*").getRegex(),
  text: me(ca.text).replace("\\b_", "\\b_| {2,}\\n").replace(/\{2,\}/g, "*").getRegex()
}, Cr = {
  normal: ja,
  gfm: sh,
  pedantic: oh
}, G0 = {
  normal: Ya,
  gfm: ca,
  breaks: kh,
  pedantic: xh
};
class Rt {
  constructor(e) {
    _e(this, "tokens");
    _e(this, "options");
    _e(this, "state");
    _e(this, "tokenizer");
    _e(this, "inlineQueue");
    this.tokens = [], this.tokens.links = /* @__PURE__ */ Object.create(null), this.options = e || k0, this.options.tokenizer = this.options.tokenizer || new tn(), this.tokenizer = this.options.tokenizer, this.tokenizer.options = this.options, this.tokenizer.lexer = this, this.inlineQueue = [], this.state = {
      inLink: !1,
      inRawBlock: !1,
      top: !0
    };
    const t = {
      block: Cr.normal,
      inline: G0.normal
    };
    this.options.pedantic ? (t.block = Cr.pedantic, t.inline = G0.pedantic) : this.options.gfm && (t.block = Cr.gfm, this.options.breaks ? t.inline = G0.breaks : t.inline = G0.gfm), this.tokenizer.rules = t;
  }
  /**
   * Expose Rules
   */
  static get rules() {
    return {
      block: Cr,
      inline: G0
    };
  }
  /**
   * Static Lex Method
   */
  static lex(e, t) {
    return new Rt(t).lex(e);
  }
  /**
   * Static Lex Inline Method
   */
  static lexInline(e, t) {
    return new Rt(t).inlineTokens(e);
  }
  /**
   * Preprocessing
   */
  lex(e) {
    e = e.replace(/\r\n|\r/g, `
`), this.blockTokens(e, this.tokens);
    for (let t = 0; t < this.inlineQueue.length; t++) {
      const r = this.inlineQueue[t];
      this.inlineTokens(r.src, r.tokens);
    }
    return this.inlineQueue = [], this.tokens;
  }
  blockTokens(e, t = []) {
    this.options.pedantic ? e = e.replace(/\t/g, "    ").replace(/^ +$/gm, "") : e = e.replace(/^( *)(\t+)/gm, (s, u, h) => u + "    ".repeat(h.length));
    let r, a, i, l;
    for (; e; )
      if (!(this.options.extensions && this.options.extensions.block && this.options.extensions.block.some((s) => (r = s.call({ lexer: this }, e, t)) ? (e = e.substring(r.raw.length), t.push(r), !0) : !1))) {
        if (r = this.tokenizer.space(e)) {
          e = e.substring(r.raw.length), r.raw.length === 1 && t.length > 0 ? t[t.length - 1].raw += `
` : t.push(r);
          continue;
        }
        if (r = this.tokenizer.code(e)) {
          e = e.substring(r.raw.length), a = t[t.length - 1], a && (a.type === "paragraph" || a.type === "text") ? (a.raw += `
` + r.raw, a.text += `
` + r.text, this.inlineQueue[this.inlineQueue.length - 1].src = a.text) : t.push(r);
          continue;
        }
        if (r = this.tokenizer.fences(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.heading(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.hr(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.blockquote(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.list(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.html(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.def(e)) {
          e = e.substring(r.raw.length), a = t[t.length - 1], a && (a.type === "paragraph" || a.type === "text") ? (a.raw += `
` + r.raw, a.text += `
` + r.raw, this.inlineQueue[this.inlineQueue.length - 1].src = a.text) : this.tokens.links[r.tag] || (this.tokens.links[r.tag] = {
            href: r.href,
            title: r.title
          });
          continue;
        }
        if (r = this.tokenizer.table(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.lheading(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (i = e, this.options.extensions && this.options.extensions.startBlock) {
          let s = 1 / 0;
          const u = e.slice(1);
          let h;
          this.options.extensions.startBlock.forEach((d) => {
            h = d.call({ lexer: this }, u), typeof h == "number" && h >= 0 && (s = Math.min(s, h));
          }), s < 1 / 0 && s >= 0 && (i = e.substring(0, s + 1));
        }
        if (this.state.top && (r = this.tokenizer.paragraph(i))) {
          a = t[t.length - 1], l && a.type === "paragraph" ? (a.raw += `
` + r.raw, a.text += `
` + r.text, this.inlineQueue.pop(), this.inlineQueue[this.inlineQueue.length - 1].src = a.text) : t.push(r), l = i.length !== e.length, e = e.substring(r.raw.length);
          continue;
        }
        if (r = this.tokenizer.text(e)) {
          e = e.substring(r.raw.length), a = t[t.length - 1], a && a.type === "text" ? (a.raw += `
` + r.raw, a.text += `
` + r.text, this.inlineQueue.pop(), this.inlineQueue[this.inlineQueue.length - 1].src = a.text) : t.push(r);
          continue;
        }
        if (e) {
          const s = "Infinite loop on byte: " + e.charCodeAt(0);
          if (this.options.silent) {
            console.error(s);
            break;
          } else
            throw new Error(s);
        }
      }
    return this.state.top = !0, t;
  }
  inline(e, t = []) {
    return this.inlineQueue.push({ src: e, tokens: t }), t;
  }
  /**
   * Lexing/Compiling
   */
  inlineTokens(e, t = []) {
    let r, a, i, l = e, s, u, h;
    if (this.tokens.links) {
      const d = Object.keys(this.tokens.links);
      if (d.length > 0)
        for (; (s = this.tokenizer.rules.inline.reflinkSearch.exec(l)) != null; )
          d.includes(s[0].slice(s[0].lastIndexOf("[") + 1, -1)) && (l = l.slice(0, s.index) + "[" + "a".repeat(s[0].length - 2) + "]" + l.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex));
    }
    for (; (s = this.tokenizer.rules.inline.blockSkip.exec(l)) != null; )
      l = l.slice(0, s.index) + "[" + "a".repeat(s[0].length - 2) + "]" + l.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);
    for (; (s = this.tokenizer.rules.inline.anyPunctuation.exec(l)) != null; )
      l = l.slice(0, s.index) + "++" + l.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);
    for (; e; )
      if (u || (h = ""), u = !1, !(this.options.extensions && this.options.extensions.inline && this.options.extensions.inline.some((d) => (r = d.call({ lexer: this }, e, t)) ? (e = e.substring(r.raw.length), t.push(r), !0) : !1))) {
        if (r = this.tokenizer.escape(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.tag(e)) {
          e = e.substring(r.raw.length), a = t[t.length - 1], a && r.type === "text" && a.type === "text" ? (a.raw += r.raw, a.text += r.text) : t.push(r);
          continue;
        }
        if (r = this.tokenizer.link(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.reflink(e, this.tokens.links)) {
          e = e.substring(r.raw.length), a = t[t.length - 1], a && r.type === "text" && a.type === "text" ? (a.raw += r.raw, a.text += r.text) : t.push(r);
          continue;
        }
        if (r = this.tokenizer.emStrong(e, l, h)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.codespan(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.br(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.del(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (r = this.tokenizer.autolink(e)) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (!this.state.inLink && (r = this.tokenizer.url(e))) {
          e = e.substring(r.raw.length), t.push(r);
          continue;
        }
        if (i = e, this.options.extensions && this.options.extensions.startInline) {
          let d = 1 / 0;
          const g = e.slice(1);
          let p;
          this.options.extensions.startInline.forEach((v) => {
            p = v.call({ lexer: this }, g), typeof p == "number" && p >= 0 && (d = Math.min(d, p));
          }), d < 1 / 0 && d >= 0 && (i = e.substring(0, d + 1));
        }
        if (r = this.tokenizer.inlineText(i)) {
          e = e.substring(r.raw.length), r.raw.slice(-1) !== "_" && (h = r.raw.slice(-1)), u = !0, a = t[t.length - 1], a && a.type === "text" ? (a.raw += r.raw, a.text += r.text) : t.push(r);
          continue;
        }
        if (e) {
          const d = "Infinite loop on byte: " + e.charCodeAt(0);
          if (this.options.silent) {
            console.error(d);
            break;
          } else
            throw new Error(d);
        }
      }
    return t;
  }
}
class nn {
  constructor(e) {
    _e(this, "options");
    this.options = e || k0;
  }
  code(e, t, r) {
    var i;
    const a = (i = (t || "").match(/^\S*/)) == null ? void 0 : i[0];
    return e = e.replace(/\n$/, "") + `
`, a ? '<pre><code class="language-' + et(a) + '">' + (r ? e : et(e, !0)) + `</code></pre>
` : "<pre><code>" + (r ? e : et(e, !0)) + `</code></pre>
`;
  }
  blockquote(e) {
    return `<blockquote>
${e}</blockquote>
`;
  }
  html(e, t) {
    return e;
  }
  heading(e, t, r) {
    return `<h${t}>${e}</h${t}>
`;
  }
  hr() {
    return `<hr>
`;
  }
  list(e, t, r) {
    const a = t ? "ol" : "ul", i = t && r !== 1 ? ' start="' + r + '"' : "";
    return "<" + a + i + `>
` + e + "</" + a + `>
`;
  }
  listitem(e, t, r) {
    return `<li>${e}</li>
`;
  }
  checkbox(e) {
    return "<input " + (e ? 'checked="" ' : "") + 'disabled="" type="checkbox">';
  }
  paragraph(e) {
    return `<p>${e}</p>
`;
  }
  table(e, t) {
    return t && (t = `<tbody>${t}</tbody>`), `<table>
<thead>
` + e + `</thead>
` + t + `</table>
`;
  }
  tablerow(e) {
    return `<tr>
${e}</tr>
`;
  }
  tablecell(e, t) {
    const r = t.header ? "th" : "td";
    return (t.align ? `<${r} align="${t.align}">` : `<${r}>`) + e + `</${r}>
`;
  }
  /**
   * span level renderer
   */
  strong(e) {
    return `<strong>${e}</strong>`;
  }
  em(e) {
    return `<em>${e}</em>`;
  }
  codespan(e) {
    return `<code>${e}</code>`;
  }
  br() {
    return "<br>";
  }
  del(e) {
    return `<del>${e}</del>`;
  }
  link(e, t, r) {
    const a = ll(e);
    if (a === null)
      return r;
    e = a;
    let i = '<a href="' + e + '"';
    return t && (i += ' title="' + t + '"'), i += ">" + r + "</a>", i;
  }
  image(e, t, r) {
    const a = ll(e);
    if (a === null)
      return r;
    e = a;
    let i = `<img src="${e}" alt="${r}"`;
    return t && (i += ` title="${t}"`), i += ">", i;
  }
  text(e) {
    return e;
  }
}
class Xa {
  // no need for block level renderers
  strong(e) {
    return e;
  }
  em(e) {
    return e;
  }
  codespan(e) {
    return e;
  }
  del(e) {
    return e;
  }
  html(e) {
    return e;
  }
  text(e) {
    return e;
  }
  link(e, t, r) {
    return "" + r;
  }
  image(e, t, r) {
    return "" + r;
  }
  br() {
    return "";
  }
}
class Nt {
  constructor(e) {
    _e(this, "options");
    _e(this, "renderer");
    _e(this, "textRenderer");
    this.options = e || k0, this.options.renderer = this.options.renderer || new nn(), this.renderer = this.options.renderer, this.renderer.options = this.options, this.textRenderer = new Xa();
  }
  /**
   * Static Parse Method
   */
  static parse(e, t) {
    return new Nt(t).parse(e);
  }
  /**
   * Static Parse Inline Method
   */
  static parseInline(e, t) {
    return new Nt(t).parseInline(e);
  }
  /**
   * Parse Loop
   */
  parse(e, t = !0) {
    let r = "";
    for (let a = 0; a < e.length; a++) {
      const i = e[a];
      if (this.options.extensions && this.options.extensions.renderers && this.options.extensions.renderers[i.type]) {
        const l = i, s = this.options.extensions.renderers[l.type].call({ parser: this }, l);
        if (s !== !1 || !["space", "hr", "heading", "code", "table", "blockquote", "list", "html", "paragraph", "text"].includes(l.type)) {
          r += s || "";
          continue;
        }
      }
      switch (i.type) {
        case "space":
          continue;
        case "hr": {
          r += this.renderer.hr();
          continue;
        }
        case "heading": {
          const l = i;
          r += this.renderer.heading(this.parseInline(l.tokens), l.depth, Yc(this.parseInline(l.tokens, this.textRenderer)));
          continue;
        }
        case "code": {
          const l = i;
          r += this.renderer.code(l.text, l.lang, !!l.escaped);
          continue;
        }
        case "table": {
          const l = i;
          let s = "", u = "";
          for (let d = 0; d < l.header.length; d++)
            u += this.renderer.tablecell(this.parseInline(l.header[d].tokens), { header: !0, align: l.align[d] });
          s += this.renderer.tablerow(u);
          let h = "";
          for (let d = 0; d < l.rows.length; d++) {
            const g = l.rows[d];
            u = "";
            for (let p = 0; p < g.length; p++)
              u += this.renderer.tablecell(this.parseInline(g[p].tokens), { header: !1, align: l.align[p] });
            h += this.renderer.tablerow(u);
          }
          r += this.renderer.table(s, h);
          continue;
        }
        case "blockquote": {
          const l = i, s = this.parse(l.tokens);
          r += this.renderer.blockquote(s);
          continue;
        }
        case "list": {
          const l = i, s = l.ordered, u = l.start, h = l.loose;
          let d = "";
          for (let g = 0; g < l.items.length; g++) {
            const p = l.items[g], v = p.checked, k = p.task;
            let A = "";
            if (p.task) {
              const C = this.renderer.checkbox(!!v);
              h ? p.tokens.length > 0 && p.tokens[0].type === "paragraph" ? (p.tokens[0].text = C + " " + p.tokens[0].text, p.tokens[0].tokens && p.tokens[0].tokens.length > 0 && p.tokens[0].tokens[0].type === "text" && (p.tokens[0].tokens[0].text = C + " " + p.tokens[0].tokens[0].text)) : p.tokens.unshift({
                type: "text",
                text: C + " "
              }) : A += C + " ";
            }
            A += this.parse(p.tokens, h), d += this.renderer.listitem(A, k, !!v);
          }
          r += this.renderer.list(d, s, u);
          continue;
        }
        case "html": {
          const l = i;
          r += this.renderer.html(l.text, l.block);
          continue;
        }
        case "paragraph": {
          const l = i;
          r += this.renderer.paragraph(this.parseInline(l.tokens));
          continue;
        }
        case "text": {
          let l = i, s = l.tokens ? this.parseInline(l.tokens) : l.text;
          for (; a + 1 < e.length && e[a + 1].type === "text"; )
            l = e[++a], s += `
` + (l.tokens ? this.parseInline(l.tokens) : l.text);
          r += t ? this.renderer.paragraph(s) : s;
          continue;
        }
        default: {
          const l = 'Token with "' + i.type + '" type was not found.';
          if (this.options.silent)
            return console.error(l), "";
          throw new Error(l);
        }
      }
    }
    return r;
  }
  /**
   * Parse Inline Tokens
   */
  parseInline(e, t) {
    t = t || this.renderer;
    let r = "";
    for (let a = 0; a < e.length; a++) {
      const i = e[a];
      if (this.options.extensions && this.options.extensions.renderers && this.options.extensions.renderers[i.type]) {
        const l = this.options.extensions.renderers[i.type].call({ parser: this }, i);
        if (l !== !1 || !["escape", "html", "link", "image", "strong", "em", "codespan", "br", "del", "text"].includes(i.type)) {
          r += l || "";
          continue;
        }
      }
      switch (i.type) {
        case "escape": {
          const l = i;
          r += t.text(l.text);
          break;
        }
        case "html": {
          const l = i;
          r += t.html(l.text);
          break;
        }
        case "link": {
          const l = i;
          r += t.link(l.href, l.title, this.parseInline(l.tokens, t));
          break;
        }
        case "image": {
          const l = i;
          r += t.image(l.href, l.title, l.text);
          break;
        }
        case "strong": {
          const l = i;
          r += t.strong(this.parseInline(l.tokens, t));
          break;
        }
        case "em": {
          const l = i;
          r += t.em(this.parseInline(l.tokens, t));
          break;
        }
        case "codespan": {
          const l = i;
          r += t.codespan(l.text);
          break;
        }
        case "br": {
          r += t.br();
          break;
        }
        case "del": {
          const l = i;
          r += t.del(this.parseInline(l.tokens, t));
          break;
        }
        case "text": {
          const l = i;
          r += t.text(l.text);
          break;
        }
        default: {
          const l = 'Token with "' + i.type + '" type was not found.';
          if (this.options.silent)
            return console.error(l), "";
          throw new Error(l);
        }
      }
    }
    return r;
  }
}
class nr {
  constructor(e) {
    _e(this, "options");
    this.options = e || k0;
  }
  /**
   * Process markdown before marked
   */
  preprocess(e) {
    return e;
  }
  /**
   * Process HTML after marked is finished
   */
  postprocess(e) {
    return e;
  }
  /**
   * Process all tokens before walk tokens
   */
  processAllTokens(e) {
    return e;
  }
}
_e(nr, "passThroughHooks", /* @__PURE__ */ new Set([
  "preprocess",
  "postprocess",
  "processAllTokens"
]));
var w0, ha, No;
class Ro {
  constructor(...e) {
    yi(this, w0);
    _e(this, "defaults", Ua());
    _e(this, "options", this.setOptions);
    _e(this, "parse", _r(this, w0, ha).call(this, Rt.lex, Nt.parse));
    _e(this, "parseInline", _r(this, w0, ha).call(this, Rt.lexInline, Nt.parseInline));
    _e(this, "Parser", Nt);
    _e(this, "Renderer", nn);
    _e(this, "TextRenderer", Xa);
    _e(this, "Lexer", Rt);
    _e(this, "Tokenizer", tn);
    _e(this, "Hooks", nr);
    this.use(...e);
  }
  /**
   * Run callback for every token
   */
  walkTokens(e, t) {
    var a, i;
    let r = [];
    for (const l of e)
      switch (r = r.concat(t.call(this, l)), l.type) {
        case "table": {
          const s = l;
          for (const u of s.header)
            r = r.concat(this.walkTokens(u.tokens, t));
          for (const u of s.rows)
            for (const h of u)
              r = r.concat(this.walkTokens(h.tokens, t));
          break;
        }
        case "list": {
          const s = l;
          r = r.concat(this.walkTokens(s.items, t));
          break;
        }
        default: {
          const s = l;
          (i = (a = this.defaults.extensions) == null ? void 0 : a.childTokens) != null && i[s.type] ? this.defaults.extensions.childTokens[s.type].forEach((u) => {
            const h = s[u].flat(1 / 0);
            r = r.concat(this.walkTokens(h, t));
          }) : s.tokens && (r = r.concat(this.walkTokens(s.tokens, t)));
        }
      }
    return r;
  }
  use(...e) {
    const t = this.defaults.extensions || { renderers: {}, childTokens: {} };
    return e.forEach((r) => {
      const a = { ...r };
      if (a.async = this.defaults.async || a.async || !1, r.extensions && (r.extensions.forEach((i) => {
        if (!i.name)
          throw new Error("extension name required");
        if ("renderer" in i) {
          const l = t.renderers[i.name];
          l ? t.renderers[i.name] = function(...s) {
            let u = i.renderer.apply(this, s);
            return u === !1 && (u = l.apply(this, s)), u;
          } : t.renderers[i.name] = i.renderer;
        }
        if ("tokenizer" in i) {
          if (!i.level || i.level !== "block" && i.level !== "inline")
            throw new Error("extension level must be 'block' or 'inline'");
          const l = t[i.level];
          l ? l.unshift(i.tokenizer) : t[i.level] = [i.tokenizer], i.start && (i.level === "block" ? t.startBlock ? t.startBlock.push(i.start) : t.startBlock = [i.start] : i.level === "inline" && (t.startInline ? t.startInline.push(i.start) : t.startInline = [i.start]));
        }
        "childTokens" in i && i.childTokens && (t.childTokens[i.name] = i.childTokens);
      }), a.extensions = t), r.renderer) {
        const i = this.defaults.renderer || new nn(this.defaults);
        for (const l in r.renderer) {
          if (!(l in i))
            throw new Error(`renderer '${l}' does not exist`);
          if (l === "options")
            continue;
          const s = l, u = r.renderer[s], h = i[s];
          i[s] = (...d) => {
            let g = u.apply(i, d);
            return g === !1 && (g = h.apply(i, d)), g || "";
          };
        }
        a.renderer = i;
      }
      if (r.tokenizer) {
        const i = this.defaults.tokenizer || new tn(this.defaults);
        for (const l in r.tokenizer) {
          if (!(l in i))
            throw new Error(`tokenizer '${l}' does not exist`);
          if (["options", "rules", "lexer"].includes(l))
            continue;
          const s = l, u = r.tokenizer[s], h = i[s];
          i[s] = (...d) => {
            let g = u.apply(i, d);
            return g === !1 && (g = h.apply(i, d)), g;
          };
        }
        a.tokenizer = i;
      }
      if (r.hooks) {
        const i = this.defaults.hooks || new nr();
        for (const l in r.hooks) {
          if (!(l in i))
            throw new Error(`hook '${l}' does not exist`);
          if (l === "options")
            continue;
          const s = l, u = r.hooks[s], h = i[s];
          nr.passThroughHooks.has(l) ? i[s] = (d) => {
            if (this.defaults.async)
              return Promise.resolve(u.call(i, d)).then((p) => h.call(i, p));
            const g = u.call(i, d);
            return h.call(i, g);
          } : i[s] = (...d) => {
            let g = u.apply(i, d);
            return g === !1 && (g = h.apply(i, d)), g;
          };
        }
        a.hooks = i;
      }
      if (r.walkTokens) {
        const i = this.defaults.walkTokens, l = r.walkTokens;
        a.walkTokens = function(s) {
          let u = [];
          return u.push(l.call(this, s)), i && (u = u.concat(i.call(this, s))), u;
        };
      }
      this.defaults = { ...this.defaults, ...a };
    }), this;
  }
  setOptions(e) {
    return this.defaults = { ...this.defaults, ...e }, this;
  }
  lexer(e, t) {
    return Rt.lex(e, t ?? this.defaults);
  }
  parser(e, t) {
    return Nt.parse(e, t ?? this.defaults);
  }
}
w0 = new WeakSet(), ha = function(e, t) {
  return (r, a) => {
    const i = { ...a }, l = { ...this.defaults, ...i };
    this.defaults.async === !0 && i.async === !1 && (l.silent || console.warn("marked(): The async option was set to true by an extension. The async: false option sent to parse will be ignored."), l.async = !0);
    const s = _r(this, w0, No).call(this, !!l.silent, !!l.async);
    if (typeof r > "u" || r === null)
      return s(new Error("marked(): input parameter is undefined or null"));
    if (typeof r != "string")
      return s(new Error("marked(): input parameter is of type " + Object.prototype.toString.call(r) + ", string expected"));
    if (l.hooks && (l.hooks.options = l), l.async)
      return Promise.resolve(l.hooks ? l.hooks.preprocess(r) : r).then((u) => e(u, l)).then((u) => l.hooks ? l.hooks.processAllTokens(u) : u).then((u) => l.walkTokens ? Promise.all(this.walkTokens(u, l.walkTokens)).then(() => u) : u).then((u) => t(u, l)).then((u) => l.hooks ? l.hooks.postprocess(u) : u).catch(s);
    try {
      l.hooks && (r = l.hooks.preprocess(r));
      let u = e(r, l);
      l.hooks && (u = l.hooks.processAllTokens(u)), l.walkTokens && this.walkTokens(u, l.walkTokens);
      let h = t(u, l);
      return l.hooks && (h = l.hooks.postprocess(h)), h;
    } catch (u) {
      return s(u);
    }
  };
}, No = function(e, t) {
  return (r) => {
    if (r.message += `
Please report this to https://github.com/markedjs/marked.`, e) {
      const a = "<p>An error occurred:</p><pre>" + et(r.message + "", !0) + "</pre>";
      return t ? Promise.resolve(a) : a;
    }
    if (t)
      return Promise.reject(r);
    throw r;
  };
};
const y0 = new Ro();
function de(n, e) {
  return y0.parse(n, e);
}
de.options = de.setOptions = function(n) {
  return y0.setOptions(n), de.defaults = y0.defaults, So(de.defaults), de;
};
de.getDefaults = Ua;
de.defaults = k0;
de.use = function(...n) {
  return y0.use(...n), de.defaults = y0.defaults, So(de.defaults), de;
};
de.walkTokens = function(n, e) {
  return y0.walkTokens(n, e);
};
de.parseInline = y0.parseInline;
de.Parser = Nt;
de.parser = Nt.parse;
de.Renderer = nn;
de.TextRenderer = Xa;
de.Lexer = Rt;
de.lexer = Rt.lex;
de.Tokenizer = tn;
de.Hooks = nr;
de.parse = de;
de.options;
de.setOptions;
de.use;
de.walkTokens;
de.parseInline;
Nt.parse;
Rt.lex;
function Dh(n) {
  if (typeof n == "function" && (n = {
    highlight: n
  }), !n || typeof n.highlight != "function")
    throw new Error("Must provide highlight function");
  return typeof n.langPrefix != "string" && (n.langPrefix = "language-"), typeof n.emptyLangClass != "string" && (n.emptyLangClass = ""), {
    async: !!n.async,
    walkTokens(e) {
      if (e.type !== "code")
        return;
      const t = cl(e.lang);
      if (n.async)
        return Promise.resolve(n.highlight(e.text, t, e.lang || "")).then(hl(e));
      const r = n.highlight(e.text, t, e.lang || "");
      if (r instanceof Promise)
        throw new Error("markedHighlight is not set to async but the highlight function is async. Set the async option to true on markedHighlight to await the async highlight function.");
      hl(e)(r);
    },
    useNewRenderer: !0,
    renderer: {
      code(e, t, r) {
        typeof e == "object" && (r = e.escaped, t = e.lang, e = e.text);
        const a = cl(t), i = a ? n.langPrefix + ml(a) : n.emptyLangClass, l = i ? ` class="${i}"` : "";
        return e = e.replace(/\n$/, ""), `<pre><code${l}>${r ? e : ml(e, !0)}
</code></pre>`;
      }
    }
  };
}
function cl(n) {
  return (n || "").match(/\S*/)[0];
}
function hl(n) {
  return (e) => {
    typeof e == "string" && e !== n.text && (n.escaped = !0, n.text = e);
  };
}
const qo = /[&<>"']/, Sh = new RegExp(qo.source, "g"), Lo = /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/, Ah = new RegExp(Lo.source, "g"), Eh = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
}, dl = (n) => Eh[n];
function ml(n, e) {
  if (e) {
    if (qo.test(n))
      return n.replace(Sh, dl);
  } else if (Lo.test(n))
    return n.replace(Ah, dl);
  return n;
}
const Fh = /[\0-\x1F!-,\.\/:-@\[-\^`\{-\xA9\xAB-\xB4\xB6-\xB9\xBB-\xBF\xD7\xF7\u02C2-\u02C5\u02D2-\u02DF\u02E5-\u02EB\u02ED\u02EF-\u02FF\u0375\u0378\u0379\u037E\u0380-\u0385\u0387\u038B\u038D\u03A2\u03F6\u0482\u0530\u0557\u0558\u055A-\u055F\u0589-\u0590\u05BE\u05C0\u05C3\u05C6\u05C8-\u05CF\u05EB-\u05EE\u05F3-\u060F\u061B-\u061F\u066A-\u066D\u06D4\u06DD\u06DE\u06E9\u06FD\u06FE\u0700-\u070F\u074B\u074C\u07B2-\u07BF\u07F6-\u07F9\u07FB\u07FC\u07FE\u07FF\u082E-\u083F\u085C-\u085F\u086B-\u089F\u08B5\u08C8-\u08D2\u08E2\u0964\u0965\u0970\u0984\u098D\u098E\u0991\u0992\u09A9\u09B1\u09B3-\u09B5\u09BA\u09BB\u09C5\u09C6\u09C9\u09CA\u09CF-\u09D6\u09D8-\u09DB\u09DE\u09E4\u09E5\u09F2-\u09FB\u09FD\u09FF\u0A00\u0A04\u0A0B-\u0A0E\u0A11\u0A12\u0A29\u0A31\u0A34\u0A37\u0A3A\u0A3B\u0A3D\u0A43-\u0A46\u0A49\u0A4A\u0A4E-\u0A50\u0A52-\u0A58\u0A5D\u0A5F-\u0A65\u0A76-\u0A80\u0A84\u0A8E\u0A92\u0AA9\u0AB1\u0AB4\u0ABA\u0ABB\u0AC6\u0ACA\u0ACE\u0ACF\u0AD1-\u0ADF\u0AE4\u0AE5\u0AF0-\u0AF8\u0B00\u0B04\u0B0D\u0B0E\u0B11\u0B12\u0B29\u0B31\u0B34\u0B3A\u0B3B\u0B45\u0B46\u0B49\u0B4A\u0B4E-\u0B54\u0B58-\u0B5B\u0B5E\u0B64\u0B65\u0B70\u0B72-\u0B81\u0B84\u0B8B-\u0B8D\u0B91\u0B96-\u0B98\u0B9B\u0B9D\u0BA0-\u0BA2\u0BA5-\u0BA7\u0BAB-\u0BAD\u0BBA-\u0BBD\u0BC3-\u0BC5\u0BC9\u0BCE\u0BCF\u0BD1-\u0BD6\u0BD8-\u0BE5\u0BF0-\u0BFF\u0C0D\u0C11\u0C29\u0C3A-\u0C3C\u0C45\u0C49\u0C4E-\u0C54\u0C57\u0C5B-\u0C5F\u0C64\u0C65\u0C70-\u0C7F\u0C84\u0C8D\u0C91\u0CA9\u0CB4\u0CBA\u0CBB\u0CC5\u0CC9\u0CCE-\u0CD4\u0CD7-\u0CDD\u0CDF\u0CE4\u0CE5\u0CF0\u0CF3-\u0CFF\u0D0D\u0D11\u0D45\u0D49\u0D4F-\u0D53\u0D58-\u0D5E\u0D64\u0D65\u0D70-\u0D79\u0D80\u0D84\u0D97-\u0D99\u0DB2\u0DBC\u0DBE\u0DBF\u0DC7-\u0DC9\u0DCB-\u0DCE\u0DD5\u0DD7\u0DE0-\u0DE5\u0DF0\u0DF1\u0DF4-\u0E00\u0E3B-\u0E3F\u0E4F\u0E5A-\u0E80\u0E83\u0E85\u0E8B\u0EA4\u0EA6\u0EBE\u0EBF\u0EC5\u0EC7\u0ECE\u0ECF\u0EDA\u0EDB\u0EE0-\u0EFF\u0F01-\u0F17\u0F1A-\u0F1F\u0F2A-\u0F34\u0F36\u0F38\u0F3A-\u0F3D\u0F48\u0F6D-\u0F70\u0F85\u0F98\u0FBD-\u0FC5\u0FC7-\u0FFF\u104A-\u104F\u109E\u109F\u10C6\u10C8-\u10CC\u10CE\u10CF\u10FB\u1249\u124E\u124F\u1257\u1259\u125E\u125F\u1289\u128E\u128F\u12B1\u12B6\u12B7\u12BF\u12C1\u12C6\u12C7\u12D7\u1311\u1316\u1317\u135B\u135C\u1360-\u137F\u1390-\u139F\u13F6\u13F7\u13FE-\u1400\u166D\u166E\u1680\u169B-\u169F\u16EB-\u16ED\u16F9-\u16FF\u170D\u1715-\u171F\u1735-\u173F\u1754-\u175F\u176D\u1771\u1774-\u177F\u17D4-\u17D6\u17D8-\u17DB\u17DE\u17DF\u17EA-\u180A\u180E\u180F\u181A-\u181F\u1879-\u187F\u18AB-\u18AF\u18F6-\u18FF\u191F\u192C-\u192F\u193C-\u1945\u196E\u196F\u1975-\u197F\u19AC-\u19AF\u19CA-\u19CF\u19DA-\u19FF\u1A1C-\u1A1F\u1A5F\u1A7D\u1A7E\u1A8A-\u1A8F\u1A9A-\u1AA6\u1AA8-\u1AAF\u1AC1-\u1AFF\u1B4C-\u1B4F\u1B5A-\u1B6A\u1B74-\u1B7F\u1BF4-\u1BFF\u1C38-\u1C3F\u1C4A-\u1C4C\u1C7E\u1C7F\u1C89-\u1C8F\u1CBB\u1CBC\u1CC0-\u1CCF\u1CD3\u1CFB-\u1CFF\u1DFA\u1F16\u1F17\u1F1E\u1F1F\u1F46\u1F47\u1F4E\u1F4F\u1F58\u1F5A\u1F5C\u1F5E\u1F7E\u1F7F\u1FB5\u1FBD\u1FBF-\u1FC1\u1FC5\u1FCD-\u1FCF\u1FD4\u1FD5\u1FDC-\u1FDF\u1FED-\u1FF1\u1FF5\u1FFD-\u203E\u2041-\u2053\u2055-\u2070\u2072-\u207E\u2080-\u208F\u209D-\u20CF\u20F1-\u2101\u2103-\u2106\u2108\u2109\u2114\u2116-\u2118\u211E-\u2123\u2125\u2127\u2129\u212E\u213A\u213B\u2140-\u2144\u214A-\u214D\u214F-\u215F\u2189-\u24B5\u24EA-\u2BFF\u2C2F\u2C5F\u2CE5-\u2CEA\u2CF4-\u2CFF\u2D26\u2D28-\u2D2C\u2D2E\u2D2F\u2D68-\u2D6E\u2D70-\u2D7E\u2D97-\u2D9F\u2DA7\u2DAF\u2DB7\u2DBF\u2DC7\u2DCF\u2DD7\u2DDF\u2E00-\u2E2E\u2E30-\u3004\u3008-\u3020\u3030\u3036\u3037\u303D-\u3040\u3097\u3098\u309B\u309C\u30A0\u30FB\u3100-\u3104\u3130\u318F-\u319F\u31C0-\u31EF\u3200-\u33FF\u4DC0-\u4DFF\u9FFD-\u9FFF\uA48D-\uA4CF\uA4FE\uA4FF\uA60D-\uA60F\uA62C-\uA63F\uA673\uA67E\uA6F2-\uA716\uA720\uA721\uA789\uA78A\uA7C0\uA7C1\uA7CB-\uA7F4\uA828-\uA82B\uA82D-\uA83F\uA874-\uA87F\uA8C6-\uA8CF\uA8DA-\uA8DF\uA8F8-\uA8FA\uA8FC\uA92E\uA92F\uA954-\uA95F\uA97D-\uA97F\uA9C1-\uA9CE\uA9DA-\uA9DF\uA9FF\uAA37-\uAA3F\uAA4E\uAA4F\uAA5A-\uAA5F\uAA77-\uAA79\uAAC3-\uAADA\uAADE\uAADF\uAAF0\uAAF1\uAAF7-\uAB00\uAB07\uAB08\uAB0F\uAB10\uAB17-\uAB1F\uAB27\uAB2F\uAB5B\uAB6A-\uAB6F\uABEB\uABEE\uABEF\uABFA-\uABFF\uD7A4-\uD7AF\uD7C7-\uD7CA\uD7FC-\uD7FF\uE000-\uF8FF\uFA6E\uFA6F\uFADA-\uFAFF\uFB07-\uFB12\uFB18-\uFB1C\uFB29\uFB37\uFB3D\uFB3F\uFB42\uFB45\uFBB2-\uFBD2\uFD3E-\uFD4F\uFD90\uFD91\uFDC8-\uFDEF\uFDFC-\uFDFF\uFE10-\uFE1F\uFE30-\uFE32\uFE35-\uFE4C\uFE50-\uFE6F\uFE75\uFEFD-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF3E\uFF40\uFF5B-\uFF65\uFFBF-\uFFC1\uFFC8\uFFC9\uFFD0\uFFD1\uFFD8\uFFD9\uFFDD-\uFFFF]|\uD800[\uDC0C\uDC27\uDC3B\uDC3E\uDC4E\uDC4F\uDC5E-\uDC7F\uDCFB-\uDD3F\uDD75-\uDDFC\uDDFE-\uDE7F\uDE9D-\uDE9F\uDED1-\uDEDF\uDEE1-\uDEFF\uDF20-\uDF2C\uDF4B-\uDF4F\uDF7B-\uDF7F\uDF9E\uDF9F\uDFC4-\uDFC7\uDFD0\uDFD6-\uDFFF]|\uD801[\uDC9E\uDC9F\uDCAA-\uDCAF\uDCD4-\uDCD7\uDCFC-\uDCFF\uDD28-\uDD2F\uDD64-\uDDFF\uDF37-\uDF3F\uDF56-\uDF5F\uDF68-\uDFFF]|\uD802[\uDC06\uDC07\uDC09\uDC36\uDC39-\uDC3B\uDC3D\uDC3E\uDC56-\uDC5F\uDC77-\uDC7F\uDC9F-\uDCDF\uDCF3\uDCF6-\uDCFF\uDD16-\uDD1F\uDD3A-\uDD7F\uDDB8-\uDDBD\uDDC0-\uDDFF\uDE04\uDE07-\uDE0B\uDE14\uDE18\uDE36\uDE37\uDE3B-\uDE3E\uDE40-\uDE5F\uDE7D-\uDE7F\uDE9D-\uDEBF\uDEC8\uDEE7-\uDEFF\uDF36-\uDF3F\uDF56-\uDF5F\uDF73-\uDF7F\uDF92-\uDFFF]|\uD803[\uDC49-\uDC7F\uDCB3-\uDCBF\uDCF3-\uDCFF\uDD28-\uDD2F\uDD3A-\uDE7F\uDEAA\uDEAD-\uDEAF\uDEB2-\uDEFF\uDF1D-\uDF26\uDF28-\uDF2F\uDF51-\uDFAF\uDFC5-\uDFDF\uDFF7-\uDFFF]|\uD804[\uDC47-\uDC65\uDC70-\uDC7E\uDCBB-\uDCCF\uDCE9-\uDCEF\uDCFA-\uDCFF\uDD35\uDD40-\uDD43\uDD48-\uDD4F\uDD74\uDD75\uDD77-\uDD7F\uDDC5-\uDDC8\uDDCD\uDDDB\uDDDD-\uDDFF\uDE12\uDE38-\uDE3D\uDE3F-\uDE7F\uDE87\uDE89\uDE8E\uDE9E\uDEA9-\uDEAF\uDEEB-\uDEEF\uDEFA-\uDEFF\uDF04\uDF0D\uDF0E\uDF11\uDF12\uDF29\uDF31\uDF34\uDF3A\uDF45\uDF46\uDF49\uDF4A\uDF4E\uDF4F\uDF51-\uDF56\uDF58-\uDF5C\uDF64\uDF65\uDF6D-\uDF6F\uDF75-\uDFFF]|\uD805[\uDC4B-\uDC4F\uDC5A-\uDC5D\uDC62-\uDC7F\uDCC6\uDCC8-\uDCCF\uDCDA-\uDD7F\uDDB6\uDDB7\uDDC1-\uDDD7\uDDDE-\uDDFF\uDE41-\uDE43\uDE45-\uDE4F\uDE5A-\uDE7F\uDEB9-\uDEBF\uDECA-\uDEFF\uDF1B\uDF1C\uDF2C-\uDF2F\uDF3A-\uDFFF]|\uD806[\uDC3B-\uDC9F\uDCEA-\uDCFE\uDD07\uDD08\uDD0A\uDD0B\uDD14\uDD17\uDD36\uDD39\uDD3A\uDD44-\uDD4F\uDD5A-\uDD9F\uDDA8\uDDA9\uDDD8\uDDD9\uDDE2\uDDE5-\uDDFF\uDE3F-\uDE46\uDE48-\uDE4F\uDE9A-\uDE9C\uDE9E-\uDEBF\uDEF9-\uDFFF]|\uD807[\uDC09\uDC37\uDC41-\uDC4F\uDC5A-\uDC71\uDC90\uDC91\uDCA8\uDCB7-\uDCFF\uDD07\uDD0A\uDD37-\uDD39\uDD3B\uDD3E\uDD48-\uDD4F\uDD5A-\uDD5F\uDD66\uDD69\uDD8F\uDD92\uDD99-\uDD9F\uDDAA-\uDEDF\uDEF7-\uDFAF\uDFB1-\uDFFF]|\uD808[\uDF9A-\uDFFF]|\uD809[\uDC6F-\uDC7F\uDD44-\uDFFF]|[\uD80A\uD80B\uD80E-\uD810\uD812-\uD819\uD824-\uD82B\uD82D\uD82E\uD830-\uD833\uD837\uD839\uD83D\uD83F\uD87B-\uD87D\uD87F\uD885-\uDB3F\uDB41-\uDBFF][\uDC00-\uDFFF]|\uD80D[\uDC2F-\uDFFF]|\uD811[\uDE47-\uDFFF]|\uD81A[\uDE39-\uDE3F\uDE5F\uDE6A-\uDECF\uDEEE\uDEEF\uDEF5-\uDEFF\uDF37-\uDF3F\uDF44-\uDF4F\uDF5A-\uDF62\uDF78-\uDF7C\uDF90-\uDFFF]|\uD81B[\uDC00-\uDE3F\uDE80-\uDEFF\uDF4B-\uDF4E\uDF88-\uDF8E\uDFA0-\uDFDF\uDFE2\uDFE5-\uDFEF\uDFF2-\uDFFF]|\uD821[\uDFF8-\uDFFF]|\uD823[\uDCD6-\uDCFF\uDD09-\uDFFF]|\uD82C[\uDD1F-\uDD4F\uDD53-\uDD63\uDD68-\uDD6F\uDEFC-\uDFFF]|\uD82F[\uDC6B-\uDC6F\uDC7D-\uDC7F\uDC89-\uDC8F\uDC9A-\uDC9C\uDC9F-\uDFFF]|\uD834[\uDC00-\uDD64\uDD6A-\uDD6C\uDD73-\uDD7A\uDD83\uDD84\uDD8C-\uDDA9\uDDAE-\uDE41\uDE45-\uDFFF]|\uD835[\uDC55\uDC9D\uDCA0\uDCA1\uDCA3\uDCA4\uDCA7\uDCA8\uDCAD\uDCBA\uDCBC\uDCC4\uDD06\uDD0B\uDD0C\uDD15\uDD1D\uDD3A\uDD3F\uDD45\uDD47-\uDD49\uDD51\uDEA6\uDEA7\uDEC1\uDEDB\uDEFB\uDF15\uDF35\uDF4F\uDF6F\uDF89\uDFA9\uDFC3\uDFCC\uDFCD]|\uD836[\uDC00-\uDDFF\uDE37-\uDE3A\uDE6D-\uDE74\uDE76-\uDE83\uDE85-\uDE9A\uDEA0\uDEB0-\uDFFF]|\uD838[\uDC07\uDC19\uDC1A\uDC22\uDC25\uDC2B-\uDCFF\uDD2D-\uDD2F\uDD3E\uDD3F\uDD4A-\uDD4D\uDD4F-\uDEBF\uDEFA-\uDFFF]|\uD83A[\uDCC5-\uDCCF\uDCD7-\uDCFF\uDD4C-\uDD4F\uDD5A-\uDFFF]|\uD83B[\uDC00-\uDDFF\uDE04\uDE20\uDE23\uDE25\uDE26\uDE28\uDE33\uDE38\uDE3A\uDE3C-\uDE41\uDE43-\uDE46\uDE48\uDE4A\uDE4C\uDE50\uDE53\uDE55\uDE56\uDE58\uDE5A\uDE5C\uDE5E\uDE60\uDE63\uDE65\uDE66\uDE6B\uDE73\uDE78\uDE7D\uDE7F\uDE8A\uDE9C-\uDEA0\uDEA4\uDEAA\uDEBC-\uDFFF]|\uD83C[\uDC00-\uDD2F\uDD4A-\uDD4F\uDD6A-\uDD6F\uDD8A-\uDFFF]|\uD83E[\uDC00-\uDFEF\uDFFA-\uDFFF]|\uD869[\uDEDE-\uDEFF]|\uD86D[\uDF35-\uDF3F]|\uD86E[\uDC1E\uDC1F]|\uD873[\uDEA2-\uDEAF]|\uD87A[\uDFE1-\uDFFF]|\uD87E[\uDE1E-\uDFFF]|\uD884[\uDF4B-\uDFFF]|\uDB40[\uDC00-\uDCFF\uDDF0-\uDFFF]/g, Ch = Object.hasOwnProperty;
class Za {
  /**
   * Create a new slug class.
   */
  constructor() {
    this.occurrences, this.reset();
  }
  /**
   * Generate a unique slug.
  *
  * Tracks previously generated slugs: repeated calls with the same value
  * will result in different slugs.
  * Use the `slug` function to get same slugs.
   *
   * @param  {string} value
   *   String of text to slugify
   * @param  {boolean} [maintainCase=false]
   *   Keep the current case, otherwise make all lowercase
   * @return {string}
   *   A unique slug string
   */
  slug(e, t) {
    const r = this;
    let a = Th(e, t === !0);
    const i = a;
    for (; Ch.call(r.occurrences, a); )
      r.occurrences[i]++, a = i + "-" + r.occurrences[i];
    return r.occurrences[a] = 0, a;
  }
  /**
   * Reset - Forget all previous slugs
   *
   * @return void
   */
  reset() {
    this.occurrences = /* @__PURE__ */ Object.create(null);
  }
}
function Th(n, e) {
  return typeof n != "string" ? "" : (e || (n = n.toLowerCase()), n.replace(Fh, "").replace(/ /g, "-"));
}
let Io = new Za(), Oo = [];
function $h({ prefix: n = "", globalSlugs: e = !1 } = {}) {
  return {
    headerIds: !1,
    // prevent deprecation warning; remove this once headerIds option is removed
    hooks: {
      preprocess(t) {
        return e || Mh(), t;
      }
    },
    renderer: {
      heading(t, r, a) {
        a = a.toLowerCase().trim().replace(/<[!\/a-z].*?>/gi, "");
        const i = `${n}${Io.slug(a)}`, l = { level: r, text: t, id: i };
        return Oo.push(l), `<h${r} id="${i}">${t}</h${r}>
`;
      }
    }
  };
}
function Mh() {
  Oo = [], Io = new Za();
}
var fl = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};
function f2(n) {
  return n && n.__esModule && Object.prototype.hasOwnProperty.call(n, "default") ? n.default : n;
}
var Po = { exports: {} };
(function(n) {
  var e = typeof window < "u" ? window : typeof WorkerGlobalScope < "u" && self instanceof WorkerGlobalScope ? self : {};
  /**
   * Prism: Lightweight, robust, elegant syntax highlighting
   *
   * @license MIT <https://opensource.org/licenses/MIT>
   * @author Lea Verou <https://lea.verou.me>
   * @namespace
   * @public
   */
  var t = function(r) {
    var a = /(?:^|\s)lang(?:uage)?-([\w-]+)(?=\s|$)/i, i = 0, l = {}, s = {
      /**
       * By default, Prism will attempt to highlight all code elements (by calling {@link Prism.highlightAll}) on the
       * current page after the page finished loading. This might be a problem if e.g. you wanted to asynchronously load
       * additional languages or plugins yourself.
       *
       * By setting this value to `true`, Prism will not automatically highlight all code elements on the page.
       *
       * You obviously have to change this value before the automatic highlighting started. To do this, you can add an
       * empty Prism object into the global scope before loading the Prism script like this:
       *
       * ```js
       * window.Prism = window.Prism || {};
       * Prism.manual = true;
       * // add a new <script> to load Prism's script
       * ```
       *
       * @default false
       * @type {boolean}
       * @memberof Prism
       * @public
       */
      manual: r.Prism && r.Prism.manual,
      /**
       * By default, if Prism is in a web worker, it assumes that it is in a worker it created itself, so it uses
       * `addEventListener` to communicate with its parent instance. However, if you're using Prism manually in your
       * own worker, you don't want it to do this.
       *
       * By setting this value to `true`, Prism will not add its own listeners to the worker.
       *
       * You obviously have to change this value before Prism executes. To do this, you can add an
       * empty Prism object into the global scope before loading the Prism script like this:
       *
       * ```js
       * window.Prism = window.Prism || {};
       * Prism.disableWorkerMessageHandler = true;
       * // Load Prism's script
       * ```
       *
       * @default false
       * @type {boolean}
       * @memberof Prism
       * @public
       */
      disableWorkerMessageHandler: r.Prism && r.Prism.disableWorkerMessageHandler,
      /**
       * A namespace for utility methods.
       *
       * All function in this namespace that are not explicitly marked as _public_ are for __internal use only__ and may
       * change or disappear at any time.
       *
       * @namespace
       * @memberof Prism
       */
      util: {
        encode: function x(_) {
          return _ instanceof u ? new u(_.type, x(_.content), _.alias) : Array.isArray(_) ? _.map(x) : _.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/\u00a0/g, " ");
        },
        /**
         * Returns the name of the type of the given value.
         *
         * @param {any} o
         * @returns {string}
         * @example
         * type(null)      === 'Null'
         * type(undefined) === 'Undefined'
         * type(123)       === 'Number'
         * type('foo')     === 'String'
         * type(true)      === 'Boolean'
         * type([1, 2])    === 'Array'
         * type({})        === 'Object'
         * type(String)    === 'Function'
         * type(/abc+/)    === 'RegExp'
         */
        type: function(x) {
          return Object.prototype.toString.call(x).slice(8, -1);
        },
        /**
         * Returns a unique number for the given object. Later calls will still return the same number.
         *
         * @param {Object} obj
         * @returns {number}
         */
        objId: function(x) {
          return x.__id || Object.defineProperty(x, "__id", { value: ++i }), x.__id;
        },
        /**
         * Creates a deep clone of the given object.
         *
         * The main intended use of this function is to clone language definitions.
         *
         * @param {T} o
         * @param {Record<number, any>} [visited]
         * @returns {T}
         * @template T
         */
        clone: function x(_, w) {
          w = w || {};
          var E, T;
          switch (s.util.type(_)) {
            case "Object":
              if (T = s.util.objId(_), w[T])
                return w[T];
              E = /** @type {Record<string, any>} */
              {}, w[T] = E;
              for (var $ in _)
                _.hasOwnProperty($) && (E[$] = x(_[$], w));
              return (
                /** @type {any} */
                E
              );
            case "Array":
              return T = s.util.objId(_), w[T] ? w[T] : (E = [], w[T] = E, /** @type {Array} */
              /** @type {any} */
              _.forEach(function(M, B) {
                E[B] = x(M, w);
              }), /** @type {any} */
              E);
            default:
              return _;
          }
        },
        /**
         * Returns the Prism language of the given element set by a `language-xxxx` or `lang-xxxx` class.
         *
         * If no language is set for the element or the element is `null` or `undefined`, `none` will be returned.
         *
         * @param {Element} element
         * @returns {string}
         */
        getLanguage: function(x) {
          for (; x; ) {
            var _ = a.exec(x.className);
            if (_)
              return _[1].toLowerCase();
            x = x.parentElement;
          }
          return "none";
        },
        /**
         * Sets the Prism `language-xxxx` class of the given element.
         *
         * @param {Element} element
         * @param {string} language
         * @returns {void}
         */
        setLanguage: function(x, _) {
          x.className = x.className.replace(RegExp(a, "gi"), ""), x.classList.add("language-" + _);
        },
        /**
         * Returns the script element that is currently executing.
         *
         * This does __not__ work for line script element.
         *
         * @returns {HTMLScriptElement | null}
         */
        currentScript: function() {
          if (typeof document > "u")
            return null;
          if ("currentScript" in document)
            return (
              /** @type {any} */
              document.currentScript
            );
          try {
            throw new Error();
          } catch (E) {
            var x = (/at [^(\r\n]*\((.*):[^:]+:[^:]+\)$/i.exec(E.stack) || [])[1];
            if (x) {
              var _ = document.getElementsByTagName("script");
              for (var w in _)
                if (_[w].src == x)
                  return _[w];
            }
            return null;
          }
        },
        /**
         * Returns whether a given class is active for `element`.
         *
         * The class can be activated if `element` or one of its ancestors has the given class and it can be deactivated
         * if `element` or one of its ancestors has the negated version of the given class. The _negated version_ of the
         * given class is just the given class with a `no-` prefix.
         *
         * Whether the class is active is determined by the closest ancestor of `element` (where `element` itself is
         * closest ancestor) that has the given class or the negated version of it. If neither `element` nor any of its
         * ancestors have the given class or the negated version of it, then the default activation will be returned.
         *
         * In the paradoxical situation where the closest ancestor contains __both__ the given class and the negated
         * version of it, the class is considered active.
         *
         * @param {Element} element
         * @param {string} className
         * @param {boolean} [defaultActivation=false]
         * @returns {boolean}
         */
        isActive: function(x, _, w) {
          for (var E = "no-" + _; x; ) {
            var T = x.classList;
            if (T.contains(_))
              return !0;
            if (T.contains(E))
              return !1;
            x = x.parentElement;
          }
          return !!w;
        }
      },
      /**
       * This namespace contains all currently loaded languages and the some helper functions to create and modify languages.
       *
       * @namespace
       * @memberof Prism
       * @public
       */
      languages: {
        /**
         * The grammar for plain, unformatted text.
         */
        plain: l,
        plaintext: l,
        text: l,
        txt: l,
        /**
         * Creates a deep copy of the language with the given id and appends the given tokens.
         *
         * If a token in `redef` also appears in the copied language, then the existing token in the copied language
         * will be overwritten at its original position.
         *
         * ## Best practices
         *
         * Since the position of overwriting tokens (token in `redef` that overwrite tokens in the copied language)
         * doesn't matter, they can technically be in any order. However, this can be confusing to others that trying to
         * understand the language definition because, normally, the order of tokens matters in Prism grammars.
         *
         * Therefore, it is encouraged to order overwriting tokens according to the positions of the overwritten tokens.
         * Furthermore, all non-overwriting tokens should be placed after the overwriting ones.
         *
         * @param {string} id The id of the language to extend. This has to be a key in `Prism.languages`.
         * @param {Grammar} redef The new tokens to append.
         * @returns {Grammar} The new language created.
         * @public
         * @example
         * Prism.languages['css-with-colors'] = Prism.languages.extend('css', {
         *     // Prism.languages.css already has a 'comment' token, so this token will overwrite CSS' 'comment' token
         *     // at its original position
         *     'comment': { ... },
         *     // CSS doesn't have a 'color' token, so this token will be appended
         *     'color': /\b(?:red|green|blue)\b/
         * });
         */
        extend: function(x, _) {
          var w = s.util.clone(s.languages[x]);
          for (var E in _)
            w[E] = _[E];
          return w;
        },
        /**
         * Inserts tokens _before_ another token in a language definition or any other grammar.
         *
         * ## Usage
         *
         * This helper method makes it easy to modify existing languages. For example, the CSS language definition
         * not only defines CSS highlighting for CSS documents, but also needs to define highlighting for CSS embedded
         * in HTML through `<style>` elements. To do this, it needs to modify `Prism.languages.markup` and add the
         * appropriate tokens. However, `Prism.languages.markup` is a regular JavaScript object literal, so if you do
         * this:
         *
         * ```js
         * Prism.languages.markup.style = {
         *     // token
         * };
         * ```
         *
         * then the `style` token will be added (and processed) at the end. `insertBefore` allows you to insert tokens
         * before existing tokens. For the CSS example above, you would use it like this:
         *
         * ```js
         * Prism.languages.insertBefore('markup', 'cdata', {
         *     'style': {
         *         // token
         *     }
         * });
         * ```
         *
         * ## Special cases
         *
         * If the grammars of `inside` and `insert` have tokens with the same name, the tokens in `inside`'s grammar
         * will be ignored.
         *
         * This behavior can be used to insert tokens after `before`:
         *
         * ```js
         * Prism.languages.insertBefore('markup', 'comment', {
         *     'comment': Prism.languages.markup.comment,
         *     // tokens after 'comment'
         * });
         * ```
         *
         * ## Limitations
         *
         * The main problem `insertBefore` has to solve is iteration order. Since ES2015, the iteration order for object
         * properties is guaranteed to be the insertion order (except for integer keys) but some browsers behave
         * differently when keys are deleted and re-inserted. So `insertBefore` can't be implemented by temporarily
         * deleting properties which is necessary to insert at arbitrary positions.
         *
         * To solve this problem, `insertBefore` doesn't actually insert the given tokens into the target object.
         * Instead, it will create a new object and replace all references to the target object with the new one. This
         * can be done without temporarily deleting properties, so the iteration order is well-defined.
         *
         * However, only references that can be reached from `Prism.languages` or `insert` will be replaced. I.e. if
         * you hold the target object in a variable, then the value of the variable will not change.
         *
         * ```js
         * var oldMarkup = Prism.languages.markup;
         * var newMarkup = Prism.languages.insertBefore('markup', 'comment', { ... });
         *
         * assert(oldMarkup !== Prism.languages.markup);
         * assert(newMarkup === Prism.languages.markup);
         * ```
         *
         * @param {string} inside The property of `root` (e.g. a language id in `Prism.languages`) that contains the
         * object to be modified.
         * @param {string} before The key to insert before.
         * @param {Grammar} insert An object containing the key-value pairs to be inserted.
         * @param {Object<string, any>} [root] The object containing `inside`, i.e. the object that contains the
         * object to be modified.
         *
         * Defaults to `Prism.languages`.
         * @returns {Grammar} The new grammar object.
         * @public
         */
        insertBefore: function(x, _, w, E) {
          E = E || /** @type {any} */
          s.languages;
          var T = E[x], $ = {};
          for (var M in T)
            if (T.hasOwnProperty(M)) {
              if (M == _)
                for (var B in w)
                  w.hasOwnProperty(B) && ($[B] = w[B]);
              w.hasOwnProperty(M) || ($[M] = T[M]);
            }
          var G = E[x];
          return E[x] = $, s.languages.DFS(s.languages, function(U, j) {
            j === G && U != x && (this[U] = $);
          }), $;
        },
        // Traverse a language definition with Depth First Search
        DFS: function x(_, w, E, T) {
          T = T || {};
          var $ = s.util.objId;
          for (var M in _)
            if (_.hasOwnProperty(M)) {
              w.call(_, M, _[M], E || M);
              var B = _[M], G = s.util.type(B);
              G === "Object" && !T[$(B)] ? (T[$(B)] = !0, x(B, w, null, T)) : G === "Array" && !T[$(B)] && (T[$(B)] = !0, x(B, w, M, T));
            }
        }
      },
      plugins: {},
      /**
       * This is the most high-level function in Prism’s API.
       * It fetches all the elements that have a `.language-xxxx` class and then calls {@link Prism.highlightElement} on
       * each one of them.
       *
       * This is equivalent to `Prism.highlightAllUnder(document, async, callback)`.
       *
       * @param {boolean} [async=false] Same as in {@link Prism.highlightAllUnder}.
       * @param {HighlightCallback} [callback] Same as in {@link Prism.highlightAllUnder}.
       * @memberof Prism
       * @public
       */
      highlightAll: function(x, _) {
        s.highlightAllUnder(document, x, _);
      },
      /**
       * Fetches all the descendants of `container` that have a `.language-xxxx` class and then calls
       * {@link Prism.highlightElement} on each one of them.
       *
       * The following hooks will be run:
       * 1. `before-highlightall`
       * 2. `before-all-elements-highlight`
       * 3. All hooks of {@link Prism.highlightElement} for each element.
       *
       * @param {ParentNode} container The root element, whose descendants that have a `.language-xxxx` class will be highlighted.
       * @param {boolean} [async=false] Whether each element is to be highlighted asynchronously using Web Workers.
       * @param {HighlightCallback} [callback] An optional callback to be invoked on each element after its highlighting is done.
       * @memberof Prism
       * @public
       */
      highlightAllUnder: function(x, _, w) {
        var E = {
          callback: w,
          container: x,
          selector: 'code[class*="language-"], [class*="language-"] code, code[class*="lang-"], [class*="lang-"] code'
        };
        s.hooks.run("before-highlightall", E), E.elements = Array.prototype.slice.apply(E.container.querySelectorAll(E.selector)), s.hooks.run("before-all-elements-highlight", E);
        for (var T = 0, $; $ = E.elements[T++]; )
          s.highlightElement($, _ === !0, E.callback);
      },
      /**
       * Highlights the code inside a single element.
       *
       * The following hooks will be run:
       * 1. `before-sanity-check`
       * 2. `before-highlight`
       * 3. All hooks of {@link Prism.highlight}. These hooks will be run by an asynchronous worker if `async` is `true`.
       * 4. `before-insert`
       * 5. `after-highlight`
       * 6. `complete`
       *
       * Some the above hooks will be skipped if the element doesn't contain any text or there is no grammar loaded for
       * the element's language.
       *
       * @param {Element} element The element containing the code.
       * It must have a class of `language-xxxx` to be processed, where `xxxx` is a valid language identifier.
       * @param {boolean} [async=false] Whether the element is to be highlighted asynchronously using Web Workers
       * to improve performance and avoid blocking the UI when highlighting very large chunks of code. This option is
       * [disabled by default](https://prismjs.com/faq.html#why-is-asynchronous-highlighting-disabled-by-default).
       *
       * Note: All language definitions required to highlight the code must be included in the main `prism.js` file for
       * asynchronous highlighting to work. You can build your own bundle on the
       * [Download page](https://prismjs.com/download.html).
       * @param {HighlightCallback} [callback] An optional callback to be invoked after the highlighting is done.
       * Mostly useful when `async` is `true`, since in that case, the highlighting is done asynchronously.
       * @memberof Prism
       * @public
       */
      highlightElement: function(x, _, w) {
        var E = s.util.getLanguage(x), T = s.languages[E];
        s.util.setLanguage(x, E);
        var $ = x.parentElement;
        $ && $.nodeName.toLowerCase() === "pre" && s.util.setLanguage($, E);
        var M = x.textContent, B = {
          element: x,
          language: E,
          grammar: T,
          code: M
        };
        function G(j) {
          B.highlightedCode = j, s.hooks.run("before-insert", B), B.element.innerHTML = B.highlightedCode, s.hooks.run("after-highlight", B), s.hooks.run("complete", B), w && w.call(B.element);
        }
        if (s.hooks.run("before-sanity-check", B), $ = B.element.parentElement, $ && $.nodeName.toLowerCase() === "pre" && !$.hasAttribute("tabindex") && $.setAttribute("tabindex", "0"), !B.code) {
          s.hooks.run("complete", B), w && w.call(B.element);
          return;
        }
        if (s.hooks.run("before-highlight", B), !B.grammar) {
          G(s.util.encode(B.code));
          return;
        }
        if (_ && r.Worker) {
          var U = new Worker(s.filename);
          U.onmessage = function(j) {
            G(j.data);
          }, U.postMessage(JSON.stringify({
            language: B.language,
            code: B.code,
            immediateClose: !0
          }));
        } else
          G(s.highlight(B.code, B.grammar, B.language));
      },
      /**
       * Low-level function, only use if you know what you’re doing. It accepts a string of text as input
       * and the language definitions to use, and returns a string with the HTML produced.
       *
       * The following hooks will be run:
       * 1. `before-tokenize`
       * 2. `after-tokenize`
       * 3. `wrap`: On each {@link Token}.
       *
       * @param {string} text A string with the code to be highlighted.
       * @param {Grammar} grammar An object containing the tokens to use.
       *
       * Usually a language definition like `Prism.languages.markup`.
       * @param {string} language The name of the language definition passed to `grammar`.
       * @returns {string} The highlighted HTML.
       * @memberof Prism
       * @public
       * @example
       * Prism.highlight('var foo = true;', Prism.languages.javascript, 'javascript');
       */
      highlight: function(x, _, w) {
        var E = {
          code: x,
          grammar: _,
          language: w
        };
        if (s.hooks.run("before-tokenize", E), !E.grammar)
          throw new Error('The language "' + E.language + '" has no grammar.');
        return E.tokens = s.tokenize(E.code, E.grammar), s.hooks.run("after-tokenize", E), u.stringify(s.util.encode(E.tokens), E.language);
      },
      /**
       * This is the heart of Prism, and the most low-level function you can use. It accepts a string of text as input
       * and the language definitions to use, and returns an array with the tokenized code.
       *
       * When the language definition includes nested tokens, the function is called recursively on each of these tokens.
       *
       * This method could be useful in other contexts as well, as a very crude parser.
       *
       * @param {string} text A string with the code to be highlighted.
       * @param {Grammar} grammar An object containing the tokens to use.
       *
       * Usually a language definition like `Prism.languages.markup`.
       * @returns {TokenStream} An array of strings and tokens, a token stream.
       * @memberof Prism
       * @public
       * @example
       * let code = `var foo = 0;`;
       * let tokens = Prism.tokenize(code, Prism.languages.javascript);
       * tokens.forEach(token => {
       *     if (token instanceof Prism.Token && token.type === 'number') {
       *         console.log(`Found numeric literal: ${token.content}`);
       *     }
       * });
       */
      tokenize: function(x, _) {
        var w = _.rest;
        if (w) {
          for (var E in w)
            _[E] = w[E];
          delete _.rest;
        }
        var T = new g();
        return p(T, T.head, x), d(x, T, _, T.head, 0), k(T);
      },
      /**
       * @namespace
       * @memberof Prism
       * @public
       */
      hooks: {
        all: {},
        /**
         * Adds the given callback to the list of callbacks for the given hook.
         *
         * The callback will be invoked when the hook it is registered for is run.
         * Hooks are usually directly run by a highlight function but you can also run hooks yourself.
         *
         * One callback function can be registered to multiple hooks and the same hook multiple times.
         *
         * @param {string} name The name of the hook.
         * @param {HookCallback} callback The callback function which is given environment variables.
         * @public
         */
        add: function(x, _) {
          var w = s.hooks.all;
          w[x] = w[x] || [], w[x].push(_);
        },
        /**
         * Runs a hook invoking all registered callbacks with the given environment variables.
         *
         * Callbacks will be invoked synchronously and in the order in which they were registered.
         *
         * @param {string} name The name of the hook.
         * @param {Object<string, any>} env The environment variables of the hook passed to all callbacks registered.
         * @public
         */
        run: function(x, _) {
          var w = s.hooks.all[x];
          if (!(!w || !w.length))
            for (var E = 0, T; T = w[E++]; )
              T(_);
        }
      },
      Token: u
    };
    r.Prism = s;
    function u(x, _, w, E) {
      this.type = x, this.content = _, this.alias = w, this.length = (E || "").length | 0;
    }
    u.stringify = function x(_, w) {
      if (typeof _ == "string")
        return _;
      if (Array.isArray(_)) {
        var E = "";
        return _.forEach(function(G) {
          E += x(G, w);
        }), E;
      }
      var T = {
        type: _.type,
        content: x(_.content, w),
        tag: "span",
        classes: ["token", _.type],
        attributes: {},
        language: w
      }, $ = _.alias;
      $ && (Array.isArray($) ? Array.prototype.push.apply(T.classes, $) : T.classes.push($)), s.hooks.run("wrap", T);
      var M = "";
      for (var B in T.attributes)
        M += " " + B + '="' + (T.attributes[B] || "").replace(/"/g, "&quot;") + '"';
      return "<" + T.tag + ' class="' + T.classes.join(" ") + '"' + M + ">" + T.content + "</" + T.tag + ">";
    };
    function h(x, _, w, E) {
      x.lastIndex = _;
      var T = x.exec(w);
      if (T && E && T[1]) {
        var $ = T[1].length;
        T.index += $, T[0] = T[0].slice($);
      }
      return T;
    }
    function d(x, _, w, E, T, $) {
      for (var M in w)
        if (!(!w.hasOwnProperty(M) || !w[M])) {
          var B = w[M];
          B = Array.isArray(B) ? B : [B];
          for (var G = 0; G < B.length; ++G) {
            if ($ && $.cause == M + "," + G)
              return;
            var U = B[G], j = U.inside, oe = !!U.lookbehind, ee = !!U.greedy, ue = U.alias;
            if (ee && !U.pattern.global) {
              var fe = U.pattern.toString().match(/[imsuy]*$/)[0];
              U.pattern = RegExp(U.pattern.source, fe + "g");
            }
            for (var Ee = U.pattern || U, ne = E.next, ve = T; ne !== _.tail && !($ && ve >= $.reach); ve += ne.value.length, ne = ne.next) {
              var we = ne.value;
              if (_.length > x.length)
                return;
              if (!(we instanceof u)) {
                var N = 1, se;
                if (ee) {
                  if (se = h(Ee, ve, x, oe), !se || se.index >= x.length)
                    break;
                  var Ie = se.index, ce = se.index + se[0].length, Ce = ve;
                  for (Ce += ne.value.length; Ie >= Ce; )
                    ne = ne.next, Ce += ne.value.length;
                  if (Ce -= ne.value.length, ve = Ce, ne.value instanceof u)
                    continue;
                  for (var O = ne; O !== _.tail && (Ce < ce || typeof O.value == "string"); O = O.next)
                    N++, Ce += O.value.length;
                  N--, we = x.slice(ve, Ce), se.index -= ve;
                } else if (se = h(Ee, 0, we, oe), !se)
                  continue;
                var Ie = se.index, Oe = se[0], Ke = we.slice(0, Ie), ft = we.slice(Ie + Oe.length), pt = ve + we.length;
                $ && pt > $.reach && ($.reach = pt);
                var Gt = ne.prev;
                Ke && (Gt = p(_, Gt, Ke), ve += Ke.length), v(_, Gt, N);
                var At = new u(M, j ? s.tokenize(Oe, j) : Oe, ue, Oe);
                if (ne = p(_, Gt, At), ft && p(_, ne, ft), N > 1) {
                  var Et = {
                    cause: M + "," + G,
                    reach: pt
                  };
                  d(x, _, w, ne.prev, ve, Et), $ && Et.reach > $.reach && ($.reach = Et.reach);
                }
              }
            }
          }
        }
    }
    function g() {
      var x = { value: null, prev: null, next: null }, _ = { value: null, prev: x, next: null };
      x.next = _, this.head = x, this.tail = _, this.length = 0;
    }
    function p(x, _, w) {
      var E = _.next, T = { value: w, prev: _, next: E };
      return _.next = T, E.prev = T, x.length++, T;
    }
    function v(x, _, w) {
      for (var E = _.next, T = 0; T < w && E !== x.tail; T++)
        E = E.next;
      _.next = E, E.prev = _, x.length -= T;
    }
    function k(x) {
      for (var _ = [], w = x.head.next; w !== x.tail; )
        _.push(w.value), w = w.next;
      return _;
    }
    if (!r.document)
      return r.addEventListener && (s.disableWorkerMessageHandler || r.addEventListener("message", function(x) {
        var _ = JSON.parse(x.data), w = _.language, E = _.code, T = _.immediateClose;
        r.postMessage(s.highlight(E, s.languages[w], w)), T && r.close();
      }, !1)), s;
    var A = s.util.currentScript();
    A && (s.filename = A.src, A.hasAttribute("data-manual") && (s.manual = !0));
    function C() {
      s.manual || s.highlightAll();
    }
    if (!s.manual) {
      var z = document.readyState;
      z === "loading" || z === "interactive" && A && A.defer ? document.addEventListener("DOMContentLoaded", C) : window.requestAnimationFrame ? window.requestAnimationFrame(C) : window.setTimeout(C, 16);
    }
    return s;
  }(e);
  n.exports && (n.exports = t), typeof fl < "u" && (fl.Prism = t), t.languages.markup = {
    comment: {
      pattern: /<!--(?:(?!<!--)[\s\S])*?-->/,
      greedy: !0
    },
    prolog: {
      pattern: /<\?[\s\S]+?\?>/,
      greedy: !0
    },
    doctype: {
      // https://www.w3.org/TR/xml/#NT-doctypedecl
      pattern: /<!DOCTYPE(?:[^>"'[\]]|"[^"]*"|'[^']*')+(?:\[(?:[^<"'\]]|"[^"]*"|'[^']*'|<(?!!--)|<!--(?:[^-]|-(?!->))*-->)*\]\s*)?>/i,
      greedy: !0,
      inside: {
        "internal-subset": {
          pattern: /(^[^\[]*\[)[\s\S]+(?=\]>$)/,
          lookbehind: !0,
          greedy: !0,
          inside: null
          // see below
        },
        string: {
          pattern: /"[^"]*"|'[^']*'/,
          greedy: !0
        },
        punctuation: /^<!|>$|[[\]]/,
        "doctype-tag": /^DOCTYPE/i,
        name: /[^\s<>'"]+/
      }
    },
    cdata: {
      pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
      greedy: !0
    },
    tag: {
      pattern: /<\/?(?!\d)[^\s>\/=$<%]+(?:\s(?:\s*[^\s>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))|(?=[\s/>])))+)?\s*\/?>/,
      greedy: !0,
      inside: {
        tag: {
          pattern: /^<\/?[^\s>\/]+/,
          inside: {
            punctuation: /^<\/?/,
            namespace: /^[^\s>\/:]+:/
          }
        },
        "special-attr": [],
        "attr-value": {
          pattern: /=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+)/,
          inside: {
            punctuation: [
              {
                pattern: /^=/,
                alias: "attr-equals"
              },
              {
                pattern: /^(\s*)["']|["']$/,
                lookbehind: !0
              }
            ]
          }
        },
        punctuation: /\/?>/,
        "attr-name": {
          pattern: /[^\s>\/]+/,
          inside: {
            namespace: /^[^\s>\/:]+:/
          }
        }
      }
    },
    entity: [
      {
        pattern: /&[\da-z]{1,8};/i,
        alias: "named-entity"
      },
      /&#x?[\da-f]{1,8};/i
    ]
  }, t.languages.markup.tag.inside["attr-value"].inside.entity = t.languages.markup.entity, t.languages.markup.doctype.inside["internal-subset"].inside = t.languages.markup, t.hooks.add("wrap", function(r) {
    r.type === "entity" && (r.attributes.title = r.content.replace(/&amp;/, "&"));
  }), Object.defineProperty(t.languages.markup.tag, "addInlined", {
    /**
     * Adds an inlined language to markup.
     *
     * An example of an inlined language is CSS with `<style>` tags.
     *
     * @param {string} tagName The name of the tag that contains the inlined language. This name will be treated as
     * case insensitive.
     * @param {string} lang The language key.
     * @example
     * addInlined('style', 'css');
     */
    value: function(a, i) {
      var l = {};
      l["language-" + i] = {
        pattern: /(^<!\[CDATA\[)[\s\S]+?(?=\]\]>$)/i,
        lookbehind: !0,
        inside: t.languages[i]
      }, l.cdata = /^<!\[CDATA\[|\]\]>$/i;
      var s = {
        "included-cdata": {
          pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
          inside: l
        }
      };
      s["language-" + i] = {
        pattern: /[\s\S]+/,
        inside: t.languages[i]
      };
      var u = {};
      u[a] = {
        pattern: RegExp(/(<__[^>]*>)(?:<!\[CDATA\[(?:[^\]]|\](?!\]>))*\]\]>|(?!<!\[CDATA\[)[\s\S])*?(?=<\/__>)/.source.replace(/__/g, function() {
          return a;
        }), "i"),
        lookbehind: !0,
        greedy: !0,
        inside: s
      }, t.languages.insertBefore("markup", "cdata", u);
    }
  }), Object.defineProperty(t.languages.markup.tag, "addAttribute", {
    /**
     * Adds an pattern to highlight languages embedded in HTML attributes.
     *
     * An example of an inlined language is CSS with `style` attributes.
     *
     * @param {string} attrName The name of the tag that contains the inlined language. This name will be treated as
     * case insensitive.
     * @param {string} lang The language key.
     * @example
     * addAttribute('style', 'css');
     */
    value: function(r, a) {
      t.languages.markup.tag.inside["special-attr"].push({
        pattern: RegExp(
          /(^|["'\s])/.source + "(?:" + r + ")" + /\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))/.source,
          "i"
        ),
        lookbehind: !0,
        inside: {
          "attr-name": /^[^\s=]+/,
          "attr-value": {
            pattern: /=[\s\S]+/,
            inside: {
              value: {
                pattern: /(^=\s*(["']|(?!["'])))\S[\s\S]*(?=\2$)/,
                lookbehind: !0,
                alias: [a, "language-" + a],
                inside: t.languages[a]
              },
              punctuation: [
                {
                  pattern: /^=/,
                  alias: "attr-equals"
                },
                /"|'/
              ]
            }
          }
        }
      });
    }
  }), t.languages.html = t.languages.markup, t.languages.mathml = t.languages.markup, t.languages.svg = t.languages.markup, t.languages.xml = t.languages.extend("markup", {}), t.languages.ssml = t.languages.xml, t.languages.atom = t.languages.xml, t.languages.rss = t.languages.xml, function(r) {
    var a = /(?:"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"|'(?:\\(?:\r\n|[\s\S])|[^'\\\r\n])*')/;
    r.languages.css = {
      comment: /\/\*[\s\S]*?\*\//,
      atrule: {
        pattern: RegExp("@[\\w-](?:" + /[^;{\s"']|\s+(?!\s)/.source + "|" + a.source + ")*?" + /(?:;|(?=\s*\{))/.source),
        inside: {
          rule: /^@[\w-]+/,
          "selector-function-argument": {
            pattern: /(\bselector\s*\(\s*(?![\s)]))(?:[^()\s]|\s+(?![\s)])|\((?:[^()]|\([^()]*\))*\))+(?=\s*\))/,
            lookbehind: !0,
            alias: "selector"
          },
          keyword: {
            pattern: /(^|[^\w-])(?:and|not|only|or)(?![\w-])/,
            lookbehind: !0
          }
          // See rest below
        }
      },
      url: {
        // https://drafts.csswg.org/css-values-3/#urls
        pattern: RegExp("\\burl\\((?:" + a.source + "|" + /(?:[^\\\r\n()"']|\\[\s\S])*/.source + ")\\)", "i"),
        greedy: !0,
        inside: {
          function: /^url/i,
          punctuation: /^\(|\)$/,
          string: {
            pattern: RegExp("^" + a.source + "$"),
            alias: "url"
          }
        }
      },
      selector: {
        pattern: RegExp(`(^|[{}\\s])[^{}\\s](?:[^{};"'\\s]|\\s+(?![\\s{])|` + a.source + ")*(?=\\s*\\{)"),
        lookbehind: !0
      },
      string: {
        pattern: a,
        greedy: !0
      },
      property: {
        pattern: /(^|[^-\w\xA0-\uFFFF])(?!\s)[-_a-z\xA0-\uFFFF](?:(?!\s)[-\w\xA0-\uFFFF])*(?=\s*:)/i,
        lookbehind: !0
      },
      important: /!important\b/i,
      function: {
        pattern: /(^|[^-a-z0-9])[-a-z0-9]+(?=\()/i,
        lookbehind: !0
      },
      punctuation: /[(){};:,]/
    }, r.languages.css.atrule.inside.rest = r.languages.css;
    var i = r.languages.markup;
    i && (i.tag.addInlined("style", "css"), i.tag.addAttribute("style", "css"));
  }(t), t.languages.clike = {
    comment: [
      {
        pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
        lookbehind: !0,
        greedy: !0
      },
      {
        pattern: /(^|[^\\:])\/\/.*/,
        lookbehind: !0,
        greedy: !0
      }
    ],
    string: {
      pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
      greedy: !0
    },
    "class-name": {
      pattern: /(\b(?:class|extends|implements|instanceof|interface|new|trait)\s+|\bcatch\s+\()[\w.\\]+/i,
      lookbehind: !0,
      inside: {
        punctuation: /[.\\]/
      }
    },
    keyword: /\b(?:break|catch|continue|do|else|finally|for|function|if|in|instanceof|new|null|return|throw|try|while)\b/,
    boolean: /\b(?:false|true)\b/,
    function: /\b\w+(?=\()/,
    number: /\b0x[\da-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?/i,
    operator: /[<>]=?|[!=]=?=?|--?|\+\+?|&&?|\|\|?|[?*/~^%]/,
    punctuation: /[{}[\];(),.:]/
  }, t.languages.javascript = t.languages.extend("clike", {
    "class-name": [
      t.languages.clike["class-name"],
      {
        pattern: /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$A-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\.(?:constructor|prototype))/,
        lookbehind: !0
      }
    ],
    keyword: [
      {
        pattern: /((?:^|\})\s*)catch\b/,
        lookbehind: !0
      },
      {
        pattern: /(^|[^.]|\.\.\.\s*)\b(?:as|assert(?=\s*\{)|async(?=\s*(?:function\b|\(|[$\w\xA0-\uFFFF]|$))|await|break|case|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally(?=\s*(?:\{|$))|for|from(?=\s*(?:['"]|$))|function|(?:get|set)(?=\s*(?:[#\[$\w\xA0-\uFFFF]|$))|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)\b/,
        lookbehind: !0
      }
    ],
    // Allow for all non-ASCII characters (See http://stackoverflow.com/a/2008444)
    function: /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*(?:\.\s*(?:apply|bind|call)\s*)?\()/,
    number: {
      pattern: RegExp(
        /(^|[^\w$])/.source + "(?:" + // constant
        (/NaN|Infinity/.source + "|" + // binary integer
        /0[bB][01]+(?:_[01]+)*n?/.source + "|" + // octal integer
        /0[oO][0-7]+(?:_[0-7]+)*n?/.source + "|" + // hexadecimal integer
        /0[xX][\dA-Fa-f]+(?:_[\dA-Fa-f]+)*n?/.source + "|" + // decimal bigint
        /\d+(?:_\d+)*n/.source + "|" + // decimal number (integer or float) but no bigint
        /(?:\d+(?:_\d+)*(?:\.(?:\d+(?:_\d+)*)?)?|\.\d+(?:_\d+)*)(?:[Ee][+-]?\d+(?:_\d+)*)?/.source) + ")" + /(?![\w$])/.source
      ),
      lookbehind: !0
    },
    operator: /--|\+\+|\*\*=?|=>|&&=?|\|\|=?|[!=]==|<<=?|>>>?=?|[-+*/%&|^!=<>]=?|\.{3}|\?\?=?|\?\.?|[~:]/
  }), t.languages.javascript["class-name"][0].pattern = /(\b(?:class|extends|implements|instanceof|interface|new)\s+)[\w.\\]+/, t.languages.insertBefore("javascript", "keyword", {
    regex: {
      pattern: RegExp(
        // lookbehind
        // eslint-disable-next-line regexp/no-dupe-characters-character-class
        /((?:^|[^$\w\xA0-\uFFFF."'\])\s]|\b(?:return|yield))\s*)/.source + // Regex pattern:
        // There are 2 regex patterns here. The RegExp set notation proposal added support for nested character
        // classes if the `v` flag is present. Unfortunately, nested CCs are both context-free and incompatible
        // with the only syntax, so we have to define 2 different regex patterns.
        /\//.source + "(?:" + /(?:\[(?:[^\]\\\r\n]|\\.)*\]|\\.|[^/\\\[\r\n])+\/[dgimyus]{0,7}/.source + "|" + // `v` flag syntax. This supports 3 levels of nested character classes.
        /(?:\[(?:[^[\]\\\r\n]|\\.|\[(?:[^[\]\\\r\n]|\\.|\[(?:[^[\]\\\r\n]|\\.)*\])*\])*\]|\\.|[^/\\\[\r\n])+\/[dgimyus]{0,7}v[dgimyus]{0,7}/.source + ")" + // lookahead
        /(?=(?:\s|\/\*(?:[^*]|\*(?!\/))*\*\/)*(?:$|[\r\n,.;:})\]]|\/\/))/.source
      ),
      lookbehind: !0,
      greedy: !0,
      inside: {
        "regex-source": {
          pattern: /^(\/)[\s\S]+(?=\/[a-z]*$)/,
          lookbehind: !0,
          alias: "language-regex",
          inside: t.languages.regex
        },
        "regex-delimiter": /^\/|\/$/,
        "regex-flags": /^[a-z]+$/
      }
    },
    // This must be declared before keyword because we use "function" inside the look-forward
    "function-variable": {
      pattern: /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*[=:]\s*(?:async\s*)?(?:\bfunction\b|(?:\((?:[^()]|\([^()]*\))*\)|(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)\s*=>))/,
      alias: "function"
    },
    parameter: [
      {
        pattern: /(function(?:\s+(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)?\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\))/,
        lookbehind: !0,
        inside: t.languages.javascript
      },
      {
        pattern: /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$a-z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*=>)/i,
        lookbehind: !0,
        inside: t.languages.javascript
      },
      {
        pattern: /(\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*=>)/,
        lookbehind: !0,
        inside: t.languages.javascript
      },
      {
        pattern: /((?:\b|\s|^)(?!(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)(?![$\w\xA0-\uFFFF]))(?:(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*)\(\s*|\]\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*\{)/,
        lookbehind: !0,
        inside: t.languages.javascript
      }
    ],
    constant: /\b[A-Z](?:[A-Z_]|\dx?)*\b/
  }), t.languages.insertBefore("javascript", "string", {
    hashbang: {
      pattern: /^#!.*/,
      greedy: !0,
      alias: "comment"
    },
    "template-string": {
      pattern: /`(?:\\[\s\S]|\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}|(?!\$\{)[^\\`])*`/,
      greedy: !0,
      inside: {
        "template-punctuation": {
          pattern: /^`|`$/,
          alias: "string"
        },
        interpolation: {
          pattern: /((?:^|[^\\])(?:\\{2})*)\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}/,
          lookbehind: !0,
          inside: {
            "interpolation-punctuation": {
              pattern: /^\$\{|\}$/,
              alias: "punctuation"
            },
            rest: t.languages.javascript
          }
        },
        string: /[\s\S]+/
      }
    },
    "string-property": {
      pattern: /((?:^|[,{])[ \t]*)(["'])(?:\\(?:\r\n|[\s\S])|(?!\2)[^\\\r\n])*\2(?=\s*:)/m,
      lookbehind: !0,
      greedy: !0,
      alias: "property"
    }
  }), t.languages.insertBefore("javascript", "operator", {
    "literal-property": {
      pattern: /((?:^|[,{])[ \t]*)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*:)/m,
      lookbehind: !0,
      alias: "property"
    }
  }), t.languages.markup && (t.languages.markup.tag.addInlined("script", "javascript"), t.languages.markup.tag.addAttribute(
    /on(?:abort|blur|change|click|composition(?:end|start|update)|dblclick|error|focus(?:in|out)?|key(?:down|up)|load|mouse(?:down|enter|leave|move|out|over|up)|reset|resize|scroll|select|slotchange|submit|unload|wheel)/.source,
    "javascript"
  )), t.languages.js = t.languages.javascript, function() {
    if (typeof t > "u" || typeof document > "u")
      return;
    Element.prototype.matches || (Element.prototype.matches = Element.prototype.msMatchesSelector || Element.prototype.webkitMatchesSelector);
    var r = "Loading…", a = function(A, C) {
      return "✖ Error " + A + " while fetching file: " + C;
    }, i = "✖ Error: File does not exist or is empty", l = {
      js: "javascript",
      py: "python",
      rb: "ruby",
      ps1: "powershell",
      psm1: "powershell",
      sh: "bash",
      bat: "batch",
      h: "c",
      tex: "latex"
    }, s = "data-src-status", u = "loading", h = "loaded", d = "failed", g = "pre[data-src]:not([" + s + '="' + h + '"]):not([' + s + '="' + u + '"])';
    function p(A, C, z) {
      var x = new XMLHttpRequest();
      x.open("GET", A, !0), x.onreadystatechange = function() {
        x.readyState == 4 && (x.status < 400 && x.responseText ? C(x.responseText) : x.status >= 400 ? z(a(x.status, x.statusText)) : z(i));
      }, x.send(null);
    }
    function v(A) {
      var C = /^\s*(\d+)\s*(?:(,)\s*(?:(\d+)\s*)?)?$/.exec(A || "");
      if (C) {
        var z = Number(C[1]), x = C[2], _ = C[3];
        return x ? _ ? [z, Number(_)] : [z, void 0] : [z, z];
      }
    }
    t.hooks.add("before-highlightall", function(A) {
      A.selector += ", " + g;
    }), t.hooks.add("before-sanity-check", function(A) {
      var C = (
        /** @type {HTMLPreElement} */
        A.element
      );
      if (C.matches(g)) {
        A.code = "", C.setAttribute(s, u);
        var z = C.appendChild(document.createElement("CODE"));
        z.textContent = r;
        var x = C.getAttribute("data-src"), _ = A.language;
        if (_ === "none") {
          var w = (/\.(\w+)$/.exec(x) || [, "none"])[1];
          _ = l[w] || w;
        }
        t.util.setLanguage(z, _), t.util.setLanguage(C, _);
        var E = t.plugins.autoloader;
        E && E.loadLanguages(_), p(
          x,
          function(T) {
            C.setAttribute(s, h);
            var $ = v(C.getAttribute("data-range"));
            if ($) {
              var M = T.split(/\r\n?|\n/g), B = $[0], G = $[1] == null ? M.length : $[1];
              B < 0 && (B += M.length), B = Math.max(0, Math.min(B - 1, M.length)), G < 0 && (G += M.length), G = Math.max(0, Math.min(G, M.length)), T = M.slice(B, G).join(`
`), C.hasAttribute("data-start") || C.setAttribute("data-start", String(B + 1));
            }
            z.textContent = T, t.highlightElement(z);
          },
          function(T) {
            C.setAttribute(s, d), z.textContent = T;
          }
        );
      }
    }), t.plugins.fileHighlight = {
      /**
       * Executes the File Highlight plugin for all matching `pre` elements under the given container.
       *
       * Note: Elements which are already loaded or currently loading will not be touched by this method.
       *
       * @param {ParentNode} [container=document]
       */
      highlight: function(C) {
        for (var z = (C || document).querySelectorAll(g), x = 0, _; _ = z[x++]; )
          t.highlightElement(_);
      }
    };
    var k = !1;
    t.fileHighlight = function() {
      k || (console.warn("Prism.fileHighlight is deprecated. Use `Prism.plugins.fileHighlight.highlight` instead."), k = !0), t.plugins.fileHighlight.highlight.apply(this, arguments);
    };
  }();
})(Po);
var In = Po.exports;
Prism.languages.python = {
  comment: {
    pattern: /(^|[^\\])#.*/,
    lookbehind: !0,
    greedy: !0
  },
  "string-interpolation": {
    pattern: /(?:f|fr|rf)(?:("""|''')[\s\S]*?\1|("|')(?:\\.|(?!\2)[^\\\r\n])*\2)/i,
    greedy: !0,
    inside: {
      interpolation: {
        // "{" <expression> <optional "!s", "!r", or "!a"> <optional ":" format specifier> "}"
        pattern: /((?:^|[^{])(?:\{\{)*)\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}])+\})+\})+\}/,
        lookbehind: !0,
        inside: {
          "format-spec": {
            pattern: /(:)[^:(){}]+(?=\}$)/,
            lookbehind: !0
          },
          "conversion-option": {
            pattern: /![sra](?=[:}]$)/,
            alias: "punctuation"
          },
          rest: null
        }
      },
      string: /[\s\S]+/
    }
  },
  "triple-quoted-string": {
    pattern: /(?:[rub]|br|rb)?("""|''')[\s\S]*?\1/i,
    greedy: !0,
    alias: "string"
  },
  string: {
    pattern: /(?:[rub]|br|rb)?("|')(?:\\.|(?!\1)[^\\\r\n])*\1/i,
    greedy: !0
  },
  function: {
    pattern: /((?:^|\s)def[ \t]+)[a-zA-Z_]\w*(?=\s*\()/g,
    lookbehind: !0
  },
  "class-name": {
    pattern: /(\bclass\s+)\w+/i,
    lookbehind: !0
  },
  decorator: {
    pattern: /(^[\t ]*)@\w+(?:\.\w+)*/m,
    lookbehind: !0,
    alias: ["annotation", "punctuation"],
    inside: {
      punctuation: /\./
    }
  },
  keyword: /\b(?:_(?=\s*:)|and|as|assert|async|await|break|case|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|match|nonlocal|not|or|pass|print|raise|return|try|while|with|yield)\b/,
  builtin: /\b(?:__import__|abs|all|any|apply|ascii|basestring|bin|bool|buffer|bytearray|bytes|callable|chr|classmethod|cmp|coerce|compile|complex|delattr|dict|dir|divmod|enumerate|eval|execfile|file|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|intern|isinstance|issubclass|iter|len|list|locals|long|map|max|memoryview|min|next|object|oct|open|ord|pow|property|range|raw_input|reduce|reload|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|unichr|unicode|vars|xrange|zip)\b/,
  boolean: /\b(?:False|None|True)\b/,
  number: /\b0(?:b(?:_?[01])+|o(?:_?[0-7])+|x(?:_?[a-f0-9])+)\b|(?:\b\d+(?:_\d+)*(?:\.(?:\d+(?:_\d+)*)?)?|\B\.\d+(?:_\d+)*)(?:e[+-]?\d+(?:_\d+)*)?j?(?!\w)/i,
  operator: /[-+%=]=?|!=|:=|\*\*?=?|\/\/?=?|<[<=>]?|>[=>]?|[&|^~]/,
  punctuation: /[{}[\];(),.:]/
};
Prism.languages.python["string-interpolation"].inside.interpolation.inside.rest = Prism.languages.python;
Prism.languages.py = Prism.languages.python;
(function(n) {
  var e = /\\(?:[^a-z()[\]]|[a-z*]+)/i, t = {
    "equation-command": {
      pattern: e,
      alias: "regex"
    }
  };
  n.languages.latex = {
    comment: /%.*/,
    // the verbatim environment prints whitespace to the document
    cdata: {
      pattern: /(\\begin\{((?:lstlisting|verbatim)\*?)\})[\s\S]*?(?=\\end\{\2\})/,
      lookbehind: !0
    },
    /*
     * equations can be between $$ $$ or $ $ or \( \) or \[ \]
     * (all are multiline)
     */
    equation: [
      {
        pattern: /\$\$(?:\\[\s\S]|[^\\$])+\$\$|\$(?:\\[\s\S]|[^\\$])+\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]/,
        inside: t,
        alias: "string"
      },
      {
        pattern: /(\\begin\{((?:align|eqnarray|equation|gather|math|multline)\*?)\})[\s\S]*?(?=\\end\{\2\})/,
        lookbehind: !0,
        inside: t,
        alias: "string"
      }
    ],
    /*
     * arguments which are keywords or references are highlighted
     * as keywords
     */
    keyword: {
      pattern: /(\\(?:begin|cite|documentclass|end|label|ref|usepackage)(?:\[[^\]]+\])?\{)[^}]+(?=\})/,
      lookbehind: !0
    },
    url: {
      pattern: /(\\url\{)[^}]+(?=\})/,
      lookbehind: !0
    },
    /*
     * section or chapter headlines are highlighted as bold so that
     * they stand out more
     */
    headline: {
      pattern: /(\\(?:chapter|frametitle|paragraph|part|section|subparagraph|subsection|subsubparagraph|subsubsection|subsubsubparagraph)\*?(?:\[[^\]]+\])?\{)[^}]+(?=\})/,
      lookbehind: !0,
      alias: "class-name"
    },
    function: {
      pattern: e,
      alias: "selector"
    },
    punctuation: /[[\]{}&]/
  }, n.languages.tex = n.languages.latex, n.languages.context = n.languages.latex;
})(Prism);
(function(n) {
  var e = "\\b(?:BASH|BASHOPTS|BASH_ALIASES|BASH_ARGC|BASH_ARGV|BASH_CMDS|BASH_COMPLETION_COMPAT_DIR|BASH_LINENO|BASH_REMATCH|BASH_SOURCE|BASH_VERSINFO|BASH_VERSION|COLORTERM|COLUMNS|COMP_WORDBREAKS|DBUS_SESSION_BUS_ADDRESS|DEFAULTS_PATH|DESKTOP_SESSION|DIRSTACK|DISPLAY|EUID|GDMSESSION|GDM_LANG|GNOME_KEYRING_CONTROL|GNOME_KEYRING_PID|GPG_AGENT_INFO|GROUPS|HISTCONTROL|HISTFILE|HISTFILESIZE|HISTSIZE|HOME|HOSTNAME|HOSTTYPE|IFS|INSTANCE|JOB|LANG|LANGUAGE|LC_ADDRESS|LC_ALL|LC_IDENTIFICATION|LC_MEASUREMENT|LC_MONETARY|LC_NAME|LC_NUMERIC|LC_PAPER|LC_TELEPHONE|LC_TIME|LESSCLOSE|LESSOPEN|LINES|LOGNAME|LS_COLORS|MACHTYPE|MAILCHECK|MANDATORY_PATH|NO_AT_BRIDGE|OLDPWD|OPTERR|OPTIND|ORBIT_SOCKETDIR|OSTYPE|PAPERSIZE|PATH|PIPESTATUS|PPID|PS1|PS2|PS3|PS4|PWD|RANDOM|REPLY|SECONDS|SELINUX_INIT|SESSION|SESSIONTYPE|SESSION_MANAGER|SHELL|SHELLOPTS|SHLVL|SSH_AUTH_SOCK|TERM|UID|UPSTART_EVENTS|UPSTART_INSTANCE|UPSTART_JOB|UPSTART_SESSION|USER|WINDOWID|XAUTHORITY|XDG_CONFIG_DIRS|XDG_CURRENT_DESKTOP|XDG_DATA_DIRS|XDG_GREETER_DATA_DIR|XDG_MENU_PREFIX|XDG_RUNTIME_DIR|XDG_SEAT|XDG_SEAT_PATH|XDG_SESSION_DESKTOP|XDG_SESSION_ID|XDG_SESSION_PATH|XDG_SESSION_TYPE|XDG_VTNR|XMODIFIERS)\\b", t = {
    pattern: /(^(["']?)\w+\2)[ \t]+\S.*/,
    lookbehind: !0,
    alias: "punctuation",
    // this looks reasonably well in all themes
    inside: null
    // see below
  }, r = {
    bash: t,
    environment: {
      pattern: RegExp("\\$" + e),
      alias: "constant"
    },
    variable: [
      // [0]: Arithmetic Environment
      {
        pattern: /\$?\(\([\s\S]+?\)\)/,
        greedy: !0,
        inside: {
          // If there is a $ sign at the beginning highlight $(( and )) as variable
          variable: [
            {
              pattern: /(^\$\(\([\s\S]+)\)\)/,
              lookbehind: !0
            },
            /^\$\(\(/
          ],
          number: /\b0x[\dA-Fa-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:[Ee]-?\d+)?/,
          // Operators according to https://www.gnu.org/software/bash/manual/bashref.html#Shell-Arithmetic
          operator: /--|\+\+|\*\*=?|<<=?|>>=?|&&|\|\||[=!+\-*/%<>^&|]=?|[?~:]/,
          // If there is no $ sign at the beginning highlight (( and )) as punctuation
          punctuation: /\(\(?|\)\)?|,|;/
        }
      },
      // [1]: Command Substitution
      {
        pattern: /\$\((?:\([^)]+\)|[^()])+\)|`[^`]+`/,
        greedy: !0,
        inside: {
          variable: /^\$\(|^`|\)$|`$/
        }
      },
      // [2]: Brace expansion
      {
        pattern: /\$\{[^}]+\}/,
        greedy: !0,
        inside: {
          operator: /:[-=?+]?|[!\/]|##?|%%?|\^\^?|,,?/,
          punctuation: /[\[\]]/,
          environment: {
            pattern: RegExp("(\\{)" + e),
            lookbehind: !0,
            alias: "constant"
          }
        }
      },
      /\$(?:\w+|[#?*!@$])/
    ],
    // Escape sequences from echo and printf's manuals, and escaped quotes.
    entity: /\\(?:[abceEfnrtv\\"]|O?[0-7]{1,3}|U[0-9a-fA-F]{8}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{1,2})/
  };
  n.languages.bash = {
    shebang: {
      pattern: /^#!\s*\/.*/,
      alias: "important"
    },
    comment: {
      pattern: /(^|[^"{\\$])#.*/,
      lookbehind: !0
    },
    "function-name": [
      // a) function foo {
      // b) foo() {
      // c) function foo() {
      // but not “foo {”
      {
        // a) and c)
        pattern: /(\bfunction\s+)[\w-]+(?=(?:\s*\(?:\s*\))?\s*\{)/,
        lookbehind: !0,
        alias: "function"
      },
      {
        // b)
        pattern: /\b[\w-]+(?=\s*\(\s*\)\s*\{)/,
        alias: "function"
      }
    ],
    // Highlight variable names as variables in for and select beginnings.
    "for-or-select": {
      pattern: /(\b(?:for|select)\s+)\w+(?=\s+in\s)/,
      alias: "variable",
      lookbehind: !0
    },
    // Highlight variable names as variables in the left-hand part
    // of assignments (“=” and “+=”).
    "assign-left": {
      pattern: /(^|[\s;|&]|[<>]\()\w+(?:\.\w+)*(?=\+?=)/,
      inside: {
        environment: {
          pattern: RegExp("(^|[\\s;|&]|[<>]\\()" + e),
          lookbehind: !0,
          alias: "constant"
        }
      },
      alias: "variable",
      lookbehind: !0
    },
    // Highlight parameter names as variables
    parameter: {
      pattern: /(^|\s)-{1,2}(?:\w+:[+-]?)?\w+(?:\.\w+)*(?=[=\s]|$)/,
      alias: "variable",
      lookbehind: !0
    },
    string: [
      // Support for Here-documents https://en.wikipedia.org/wiki/Here_document
      {
        pattern: /((?:^|[^<])<<-?\s*)(\w+)\s[\s\S]*?(?:\r?\n|\r)\2/,
        lookbehind: !0,
        greedy: !0,
        inside: r
      },
      // Here-document with quotes around the tag
      // → No expansion (so no “inside”).
      {
        pattern: /((?:^|[^<])<<-?\s*)(["'])(\w+)\2\s[\s\S]*?(?:\r?\n|\r)\3/,
        lookbehind: !0,
        greedy: !0,
        inside: {
          bash: t
        }
      },
      // “Normal” string
      {
        // https://www.gnu.org/software/bash/manual/html_node/Double-Quotes.html
        pattern: /(^|[^\\](?:\\\\)*)"(?:\\[\s\S]|\$\([^)]+\)|\$(?!\()|`[^`]+`|[^"\\`$])*"/,
        lookbehind: !0,
        greedy: !0,
        inside: r
      },
      {
        // https://www.gnu.org/software/bash/manual/html_node/Single-Quotes.html
        pattern: /(^|[^$\\])'[^']*'/,
        lookbehind: !0,
        greedy: !0
      },
      {
        // https://www.gnu.org/software/bash/manual/html_node/ANSI_002dC-Quoting.html
        pattern: /\$'(?:[^'\\]|\\[\s\S])*'/,
        greedy: !0,
        inside: {
          entity: r.entity
        }
      }
    ],
    environment: {
      pattern: RegExp("\\$?" + e),
      alias: "constant"
    },
    variable: r.variable,
    function: {
      pattern: /(^|[\s;|&]|[<>]\()(?:add|apropos|apt|apt-cache|apt-get|aptitude|aspell|automysqlbackup|awk|basename|bash|bc|bconsole|bg|bzip2|cal|cargo|cat|cfdisk|chgrp|chkconfig|chmod|chown|chroot|cksum|clear|cmp|column|comm|composer|cp|cron|crontab|csplit|curl|cut|date|dc|dd|ddrescue|debootstrap|df|diff|diff3|dig|dir|dircolors|dirname|dirs|dmesg|docker|docker-compose|du|egrep|eject|env|ethtool|expand|expect|expr|fdformat|fdisk|fg|fgrep|file|find|fmt|fold|format|free|fsck|ftp|fuser|gawk|git|gparted|grep|groupadd|groupdel|groupmod|groups|grub-mkconfig|gzip|halt|head|hg|history|host|hostname|htop|iconv|id|ifconfig|ifdown|ifup|import|install|ip|java|jobs|join|kill|killall|less|link|ln|locate|logname|logrotate|look|lpc|lpr|lprint|lprintd|lprintq|lprm|ls|lsof|lynx|make|man|mc|mdadm|mkconfig|mkdir|mke2fs|mkfifo|mkfs|mkisofs|mknod|mkswap|mmv|more|most|mount|mtools|mtr|mutt|mv|nano|nc|netstat|nice|nl|node|nohup|notify-send|npm|nslookup|op|open|parted|passwd|paste|pathchk|ping|pkill|pnpm|podman|podman-compose|popd|pr|printcap|printenv|ps|pushd|pv|quota|quotacheck|quotactl|ram|rar|rcp|reboot|remsync|rename|renice|rev|rm|rmdir|rpm|rsync|scp|screen|sdiff|sed|sendmail|seq|service|sftp|sh|shellcheck|shuf|shutdown|sleep|slocate|sort|split|ssh|stat|strace|su|sudo|sum|suspend|swapon|sync|sysctl|tac|tail|tar|tee|time|timeout|top|touch|tr|traceroute|tsort|tty|umount|uname|unexpand|uniq|units|unrar|unshar|unzip|update-grub|uptime|useradd|userdel|usermod|users|uudecode|uuencode|v|vcpkg|vdir|vi|vim|virsh|vmstat|wait|watch|wc|wget|whereis|which|who|whoami|write|xargs|xdg-open|yarn|yes|zenity|zip|zsh|zypper)(?=$|[)\s;|&])/,
      lookbehind: !0
    },
    keyword: {
      pattern: /(^|[\s;|&]|[<>]\()(?:case|do|done|elif|else|esac|fi|for|function|if|in|select|then|until|while)(?=$|[)\s;|&])/,
      lookbehind: !0
    },
    // https://www.gnu.org/software/bash/manual/html_node/Shell-Builtin-Commands.html
    builtin: {
      pattern: /(^|[\s;|&]|[<>]\()(?:\.|:|alias|bind|break|builtin|caller|cd|command|continue|declare|echo|enable|eval|exec|exit|export|getopts|hash|help|let|local|logout|mapfile|printf|pwd|read|readarray|readonly|return|set|shift|shopt|source|test|times|trap|type|typeset|ulimit|umask|unalias|unset)(?=$|[)\s;|&])/,
      lookbehind: !0,
      // Alias added to make those easier to distinguish from strings.
      alias: "class-name"
    },
    boolean: {
      pattern: /(^|[\s;|&]|[<>]\()(?:false|true)(?=$|[)\s;|&])/,
      lookbehind: !0
    },
    "file-descriptor": {
      pattern: /\B&\d\b/,
      alias: "important"
    },
    operator: {
      // Lots of redirections here, but not just that.
      pattern: /\d?<>|>\||\+=|=[=~]?|!=?|<<[<-]?|[&\d]?>>|\d[<>]&?|[<>][&=]?|&[>&]?|\|[&|]?/,
      inside: {
        "file-descriptor": {
          pattern: /^\d/,
          alias: "important"
        }
      }
    },
    punctuation: /\$?\(\(?|\)\)?|\.\.|[{}[\];\\]/,
    number: {
      pattern: /(^|\s)(?:[1-9]\d*|0)(?:[.,]\d+)?\b/,
      lookbehind: !0
    }
  }, t.inside = n.languages.bash;
  for (var a = [
    "comment",
    "function-name",
    "for-or-select",
    "assign-left",
    "parameter",
    "string",
    "environment",
    "function",
    "keyword",
    "builtin",
    "boolean",
    "file-descriptor",
    "operator",
    "punctuation",
    "number"
  ], i = r.variable[1].inside, l = 0; l < a.length; l++)
    i[a[l]] = n.languages.bash[a[l]];
  n.languages.sh = n.languages.bash, n.languages.shell = n.languages.bash;
})(Prism);
const zh = '<svg class="md-link-icon" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true" fill="currentColor"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg>', Bh = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 15 15" color="currentColor" aria-hidden="true" aria-label="Copy" stroke-width="1.3" width="15" height="15">
  <path fill="currentColor" d="M12.728 4.545v8.182H4.545V4.545zm0 -0.909H4.545a0.909 0.909 0 0 0 -0.909 0.909v8.182a0.909 0.909 0 0 0 0.909 0.909h8.182a0.909 0.909 0 0 0 0.909 -0.909V4.545a0.909 0.909 0 0 0 -0.909 -0.909"/>
  <path fill="currentColor" d="M1.818 8.182H0.909V1.818a0.909 0.909 0 0 1 0.909 -0.909h6.364v0.909H1.818Z"/>
</svg>

`, Rh = `<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 17 17" aria-hidden="true" aria-label="Copied" fill="none" stroke="currentColor" stroke-width="1.3">
  <path d="m13.813 4.781 -7.438 7.438 -3.188 -3.188"/>
</svg>
`, pl = `<button title="copy" class="copy_code_button">
  <span class="copy-text">${Bh}</span>
  <span class="check">${Rh}</span>
</button>`, Ho = /[&<>"']/, Nh = new RegExp(Ho.source, "g"), Uo = /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/, qh = new RegExp(Uo.source, "g"), Lh = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
}, gl = (n) => Lh[n] || "";
function On(n, e) {
  if (e) {
    if (Ho.test(n))
      return n.replace(Nh, gl);
  } else if (Uo.test(n))
    return n.replace(qh, gl);
  return n;
}
function Ih(n) {
  const e = n.map((t) => ({
    start: new RegExp(t.left.replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&")),
    end: new RegExp(t.right.replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&"))
  }));
  return {
    name: "latex",
    level: "block",
    start(t) {
      for (const r of e) {
        const a = t.match(r.start);
        if (a)
          return a.index;
      }
      return -1;
    },
    tokenizer(t, r) {
      for (const a of e) {
        const i = new RegExp(
          `${a.start.source}([\\s\\S]+?)${a.end.source}`
        ).exec(t);
        if (i)
          return {
            type: "latex",
            raw: i[0],
            text: i[1].trim()
          };
      }
    },
    renderer(t) {
      return `<div class="latex-block">${t.text}</div>`;
    }
  };
}
function Oh() {
  return {
    name: "mermaid",
    level: "block",
    start(n) {
      var e;
      return (e = n.match(/^```mermaid\s*\n/)) == null ? void 0 : e.index;
    },
    tokenizer(n) {
      const e = /^```mermaid\s*\n([\s\S]*?)```\s*(?:\n|$)/.exec(n);
      if (e)
        return {
          type: "mermaid",
          raw: e[0],
          text: e[1].trim()
        };
    },
    renderer(n) {
      return `<div class="mermaid">${n.text}</div>
`;
    }
  };
}
const Ph = {
  code(n, e, t) {
    var a;
    const r = ((a = (e ?? "").match(/\S*/)) == null ? void 0 : a[0]) ?? "";
    return n = n.replace(/\n$/, "") + `
`, !r || r === "mermaid" ? '<div class="code_wrap">' + pl + "<pre><code>" + (t ? n : On(n, !0)) + `</code></pre></div>
` : '<div class="code_wrap">' + pl + '<pre><code class="language-' + On(r) + '">' + (t ? n : On(n, !0)) + `</code></pre></div>
`;
  }
}, Hh = new Za();
function Uh({
  header_links: n,
  line_breaks: e,
  latex_delimiters: t
}) {
  const r = new Ro();
  r.use(
    {
      gfm: !0,
      pedantic: !1,
      breaks: e
    },
    Dh({
      highlight: (l, s) => {
        var u;
        return (u = In.languages) != null && u[s] ? In.highlight(l, In.languages[s], s) : l;
      }
    }),
    { renderer: Ph }
  ), n && (r.use($h()), r.use({
    extensions: [
      {
        name: "heading",
        level: "block",
        renderer(l) {
          const s = l.raw.toLowerCase().trim().replace(/<[!\/a-z].*?>/gi, ""), u = "h" + Hh.slug(s), h = l.depth, d = this.parser.parseInline(l.tokens);
          return `<h${h} id="${u}"><a class="md-header-anchor" href="#${u}">${zh}</a>${d}</h${h}>
`;
        }
      }
    ]
  }));
  const a = Oh(), i = Ih(t);
  return r.use({
    extensions: [a, i]
  }), r;
}
const da = (n) => JSON.parse(JSON.stringify(n)), Gh = (n) => n.nodeType === 1, Vh = (n) => d4.has(n.tagName), Wh = (n) => "action" in n, jh = (n) => n.tagName === "IFRAME", Yh = (n) => "formAction" in n, Xh = (n) => "protocol" in n, Tr = /* @__PURE__ */ (() => {
  const n = /^(?:\w+script|data):/i;
  return (e) => n.test(e);
})(), Zh = /* @__PURE__ */ (() => {
  const n = /(?:script|data):/i;
  return (e) => n.test(e);
})(), Kh = (n) => {
  const e = {};
  for (let t = 0, r = n.length; t < r; t++) {
    const a = n[t];
    for (const i in a)
      e[i] ? e[i] = e[i].concat(a[i]) : e[i] = a[i];
  }
  return e;
}, Go = (n, e) => {
  let t = n.firstChild;
  for (; t; ) {
    const r = t.nextSibling;
    Gh(t) && (e(t, n), t.parentNode && Go(t, e)), t = r;
  }
}, Qh = (n, e) => {
  const t = document.createNodeIterator(n, NodeFilter.SHOW_ELEMENT);
  let r;
  for (; r = t.nextNode(); ) {
    const a = r.parentNode;
    a && e(r, a);
  }
}, Jh = (n, e) => !!globalThis.document && !!globalThis.document.createNodeIterator ? Qh(n, e) : Go(n, e), Vo = [
  "a",
  "abbr",
  "acronym",
  "address",
  "area",
  "article",
  "aside",
  "audio",
  "b",
  "bdi",
  "bdo",
  "bgsound",
  "big",
  "blockquote",
  "body",
  "br",
  "button",
  "canvas",
  "caption",
  "center",
  "cite",
  "code",
  "col",
  "colgroup",
  "datalist",
  "dd",
  "del",
  "details",
  "dfn",
  "dialog",
  "dir",
  "div",
  "dl",
  "dt",
  "em",
  "fieldset",
  "figcaption",
  "figure",
  "font",
  "footer",
  "form",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "head",
  "header",
  "hgroup",
  "hr",
  "html",
  "i",
  "img",
  "input",
  "ins",
  "kbd",
  "keygen",
  "label",
  "layer",
  "legend",
  "li",
  "link",
  "listing",
  "main",
  "map",
  "mark",
  "marquee",
  "menu",
  "meta",
  "meter",
  "nav",
  "nobr",
  "ol",
  "optgroup",
  "option",
  "output",
  "p",
  "picture",
  "popup",
  "pre",
  "progress",
  "q",
  "rb",
  "rp",
  "rt",
  "rtc",
  "ruby",
  "s",
  "samp",
  "section",
  "select",
  "selectmenu",
  "small",
  "source",
  "span",
  "strike",
  "strong",
  "style",
  "sub",
  "summary",
  "sup",
  "table",
  "tbody",
  "td",
  "tfoot",
  "th",
  "thead",
  "time",
  "tr",
  "track",
  "tt",
  "u",
  "ul",
  "var",
  "video",
  "wbr"
], e4 = [
  "basefont",
  "command",
  "data",
  "iframe",
  "image",
  "plaintext",
  "portal",
  "slot",
  // 'template', //TODO: Not exactly correct to never allow this, too strict
  "textarea",
  "title",
  "xmp"
], t4 = /* @__PURE__ */ new Set([
  ...Vo,
  ...e4
]), Wo = [
  "svg",
  "a",
  "altglyph",
  "altglyphdef",
  "altglyphitem",
  "animatecolor",
  "animatemotion",
  "animatetransform",
  "circle",
  "clippath",
  "defs",
  "desc",
  "ellipse",
  "filter",
  "font",
  "g",
  "glyph",
  "glyphref",
  "hkern",
  "image",
  "line",
  "lineargradient",
  "marker",
  "mask",
  "metadata",
  "mpath",
  "path",
  "pattern",
  "polygon",
  "polyline",
  "radialgradient",
  "rect",
  "stop",
  "style",
  "switch",
  "symbol",
  "text",
  "textpath",
  "title",
  "tref",
  "tspan",
  "view",
  "vkern",
  /* FILTERS */
  "feBlend",
  "feColorMatrix",
  "feComponentTransfer",
  "feComposite",
  "feConvolveMatrix",
  "feDiffuseLighting",
  "feDisplacementMap",
  "feDistantLight",
  "feFlood",
  "feFuncA",
  "feFuncB",
  "feFuncG",
  "feFuncR",
  "feGaussianBlur",
  "feImage",
  "feMerge",
  "feMergeNode",
  "feMorphology",
  "feOffset",
  "fePointLight",
  "feSpecularLighting",
  "feSpotLight",
  "feTile",
  "feTurbulence"
], r4 = [
  "animate",
  "color-profile",
  "cursor",
  "discard",
  "fedropshadow",
  "font-face",
  "font-face-format",
  "font-face-name",
  "font-face-src",
  "font-face-uri",
  "foreignobject",
  "hatch",
  "hatchpath",
  "mesh",
  "meshgradient",
  "meshpatch",
  "meshrow",
  "missing-glyph",
  "script",
  "set",
  "solidcolor",
  "unknown",
  "use"
], n4 = /* @__PURE__ */ new Set([
  ...Wo,
  ...r4
]), jo = [
  "math",
  "menclose",
  "merror",
  "mfenced",
  "mfrac",
  "mglyph",
  "mi",
  "mlabeledtr",
  "mmultiscripts",
  "mn",
  "mo",
  "mover",
  "mpadded",
  "mphantom",
  "mroot",
  "mrow",
  "ms",
  "mspace",
  "msqrt",
  "mstyle",
  "msub",
  "msup",
  "msubsup",
  "mtable",
  "mtd",
  "mtext",
  "mtr",
  "munder",
  "munderover"
], a4 = [
  "maction",
  "maligngroup",
  "malignmark",
  "mlongdiv",
  "mscarries",
  "mscarry",
  "msgroup",
  "mstack",
  "msline",
  "msrow",
  "semantics",
  "annotation",
  "annotation-xml",
  "mprescripts",
  "none"
], i4 = /* @__PURE__ */ new Set([
  ...jo,
  ...a4
]), l4 = [
  "abbr",
  "accept",
  "accept-charset",
  "accesskey",
  "action",
  "align",
  "alink",
  "allow",
  "allowfullscreen",
  "alt",
  "anchor",
  "archive",
  "as",
  "async",
  "autocapitalize",
  "autocomplete",
  "autocorrect",
  "autofocus",
  "autopictureinpicture",
  "autoplay",
  "axis",
  "background",
  "behavior",
  "bgcolor",
  "border",
  "bordercolor",
  "capture",
  "cellpadding",
  "cellspacing",
  "challenge",
  "char",
  "charoff",
  "charset",
  "checked",
  "cite",
  "class",
  "classid",
  "clear",
  "code",
  "codebase",
  "codetype",
  "color",
  "cols",
  "colspan",
  "compact",
  "content",
  "contenteditable",
  "controls",
  "controlslist",
  "conversiondestination",
  "coords",
  "crossorigin",
  "csp",
  "data",
  "datetime",
  "declare",
  "decoding",
  "default",
  "defer",
  "dir",
  "direction",
  "dirname",
  "disabled",
  "disablepictureinpicture",
  "disableremoteplayback",
  "disallowdocumentaccess",
  "download",
  "draggable",
  "elementtiming",
  "enctype",
  "end",
  "enterkeyhint",
  "event",
  "exportparts",
  "face",
  "for",
  "form",
  "formaction",
  "formenctype",
  "formmethod",
  "formnovalidate",
  "formtarget",
  "frame",
  "frameborder",
  "headers",
  "height",
  "hidden",
  "high",
  "href",
  "hreflang",
  "hreftranslate",
  "hspace",
  "http-equiv",
  "id",
  "imagesizes",
  "imagesrcset",
  "importance",
  "impressiondata",
  "impressionexpiry",
  "incremental",
  "inert",
  "inputmode",
  "integrity",
  "invisible",
  "ismap",
  "keytype",
  "kind",
  "label",
  "lang",
  "language",
  "latencyhint",
  "leftmargin",
  "link",
  "list",
  "loading",
  "longdesc",
  "loop",
  "low",
  "lowsrc",
  "manifest",
  "marginheight",
  "marginwidth",
  "max",
  "maxlength",
  "mayscript",
  "media",
  "method",
  "min",
  "minlength",
  "multiple",
  "muted",
  "name",
  "nohref",
  "nomodule",
  "nonce",
  "noresize",
  "noshade",
  "novalidate",
  "nowrap",
  "object",
  "open",
  "optimum",
  "part",
  "pattern",
  "ping",
  "placeholder",
  "playsinline",
  "policy",
  "poster",
  "preload",
  "pseudo",
  "readonly",
  "referrerpolicy",
  "rel",
  "reportingorigin",
  "required",
  "resources",
  "rev",
  "reversed",
  "role",
  "rows",
  "rowspan",
  "rules",
  "sandbox",
  "scheme",
  "scope",
  "scopes",
  "scrollamount",
  "scrolldelay",
  "scrolling",
  "select",
  "selected",
  "shadowroot",
  "shadowrootdelegatesfocus",
  "shape",
  "size",
  "sizes",
  "slot",
  "span",
  "spellcheck",
  "src",
  "srclang",
  "srcset",
  "standby",
  "start",
  "step",
  "style",
  "summary",
  "tabindex",
  "target",
  "text",
  "title",
  "topmargin",
  "translate",
  "truespeed",
  "trusttoken",
  "type",
  "usemap",
  "valign",
  "value",
  "valuetype",
  "version",
  "virtualkeyboardpolicy",
  "vlink",
  "vspace",
  "webkitdirectory",
  "width",
  "wrap"
], s4 = [
  "accent-height",
  "accumulate",
  "additive",
  "alignment-baseline",
  "ascent",
  "attributename",
  "attributetype",
  "azimuth",
  "basefrequency",
  "baseline-shift",
  "begin",
  "bias",
  "by",
  "class",
  "clip",
  "clippathunits",
  "clip-path",
  "clip-rule",
  "color",
  "color-interpolation",
  "color-interpolation-filters",
  "color-profile",
  "color-rendering",
  "cx",
  "cy",
  "d",
  "dx",
  "dy",
  "diffuseconstant",
  "direction",
  "display",
  "divisor",
  "dominant-baseline",
  "dur",
  "edgemode",
  "elevation",
  "end",
  "fill",
  "fill-opacity",
  "fill-rule",
  "filter",
  "filterunits",
  "flood-color",
  "flood-opacity",
  "font-family",
  "font-size",
  "font-size-adjust",
  "font-stretch",
  "font-style",
  "font-variant",
  "font-weight",
  "fx",
  "fy",
  "g1",
  "g2",
  "glyph-name",
  "glyphref",
  "gradientunits",
  "gradienttransform",
  "height",
  "href",
  "id",
  "image-rendering",
  "in",
  "in2",
  "k",
  "k1",
  "k2",
  "k3",
  "k4",
  "kerning",
  "keypoints",
  "keysplines",
  "keytimes",
  "lang",
  "lengthadjust",
  "letter-spacing",
  "kernelmatrix",
  "kernelunitlength",
  "lighting-color",
  "local",
  "marker-end",
  "marker-mid",
  "marker-start",
  "markerheight",
  "markerunits",
  "markerwidth",
  "maskcontentunits",
  "maskunits",
  "max",
  "mask",
  "media",
  "method",
  "mode",
  "min",
  "name",
  "numoctaves",
  "offset",
  "operator",
  "opacity",
  "order",
  "orient",
  "orientation",
  "origin",
  "overflow",
  "paint-order",
  "path",
  "pathlength",
  "patterncontentunits",
  "patterntransform",
  "patternunits",
  "points",
  "preservealpha",
  "preserveaspectratio",
  "primitiveunits",
  "r",
  "rx",
  "ry",
  "radius",
  "refx",
  "refy",
  "repeatcount",
  "repeatdur",
  "restart",
  "result",
  "rotate",
  "scale",
  "seed",
  "shape-rendering",
  "specularconstant",
  "specularexponent",
  "spreadmethod",
  "startoffset",
  "stddeviation",
  "stitchtiles",
  "stop-color",
  "stop-opacity",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "stroke-miterlimit",
  "stroke-opacity",
  "stroke",
  "stroke-width",
  "style",
  "surfacescale",
  "systemlanguage",
  "tabindex",
  "targetx",
  "targety",
  "transform",
  "transform-origin",
  "text-anchor",
  "text-decoration",
  "text-rendering",
  "textlength",
  "type",
  "u1",
  "u2",
  "unicode",
  "values",
  "viewbox",
  "visibility",
  "version",
  "vert-adv-y",
  "vert-origin-x",
  "vert-origin-y",
  "width",
  "word-spacing",
  "wrap",
  "writing-mode",
  "xchannelselector",
  "ychannelselector",
  "x",
  "x1",
  "x2",
  "xmlns",
  "y",
  "y1",
  "y2",
  "z",
  "zoomandpan"
], o4 = [
  "accent",
  "accentunder",
  "align",
  "bevelled",
  "close",
  "columnsalign",
  "columnlines",
  "columnspan",
  "denomalign",
  "depth",
  "dir",
  "display",
  "displaystyle",
  "encoding",
  "fence",
  "frame",
  "height",
  "href",
  "id",
  "largeop",
  "length",
  "linethickness",
  "lspace",
  "lquote",
  "mathbackground",
  "mathcolor",
  "mathsize",
  "mathvariant",
  "maxsize",
  "minsize",
  "movablelimits",
  "notation",
  "numalign",
  "open",
  "rowalign",
  "rowlines",
  "rowspacing",
  "rowspan",
  "rspace",
  "rquote",
  "scriptlevel",
  "scriptminsize",
  "scriptsizemultiplier",
  "selection",
  "separator",
  "separators",
  "stretchy",
  "subscriptshift",
  "supscriptshift",
  "symmetric",
  "voffset",
  "width",
  "xmlns"
], St = {
  HTML: "http://www.w3.org/1999/xhtml",
  SVG: "http://www.w3.org/2000/svg",
  MATH: "http://www.w3.org/1998/Math/MathML"
}, u4 = {
  [St.HTML]: t4,
  [St.SVG]: n4,
  [St.MATH]: i4
}, c4 = {
  [St.HTML]: "html",
  [St.SVG]: "svg",
  [St.MATH]: "math"
}, h4 = {
  [St.HTML]: "",
  [St.SVG]: "svg:",
  [St.MATH]: "math:"
}, d4 = /* @__PURE__ */ new Set([
  "A",
  "AREA",
  "BUTTON",
  "FORM",
  "IFRAME",
  "INPUT"
]), Yo = {
  allowComments: !0,
  allowCustomElements: !1,
  allowUnknownMarkup: !1,
  allowElements: [
    ...Vo,
    ...Wo.map((n) => `svg:${n}`),
    ...jo.map((n) => `math:${n}`)
  ],
  allowAttributes: Kh([
    Object.fromEntries(l4.map((n) => [n, ["*"]])),
    Object.fromEntries(s4.map((n) => [n, ["svg:*"]])),
    Object.fromEntries(o4.map((n) => [n, ["math:*"]]))
  ])
};
var Pn = function(n, e, t, r, a) {
  if (r === "m") throw new TypeError("Private method is not writable");
  if (r === "a" && !a) throw new TypeError("Private accessor was defined without a setter");
  if (typeof e == "function" ? n !== e || !a : !e.has(n)) throw new TypeError("Cannot write private member to an object whose class did not declare it");
  return r === "a" ? a.call(n, t) : a ? a.value = t : e.set(n, t), t;
}, _0 = function(n, e, t, r) {
  if (t === "a" && !r) throw new TypeError("Private accessor was defined without a getter");
  if (typeof e == "function" ? n !== e || !r : !e.has(n)) throw new TypeError("Cannot read private member from an object whose class did not declare it");
  return t === "m" ? r : t === "a" ? r.call(n) : r ? r.value : e.get(n);
}, u0, Ur, Gr;
class Xo {
  /* CONSTRUCTOR */
  constructor(e = {}) {
    u0.set(this, void 0), Ur.set(this, void 0), Gr.set(this, void 0), this.getConfiguration = () => da(_0(this, u0, "f")), this.sanitize = (d) => {
      const g = _0(this, Ur, "f"), p = _0(this, Gr, "f");
      return Jh(d, (v, k) => {
        const A = v.namespaceURI || St.HTML, C = k.namespaceURI || St.HTML, z = u4[A], x = c4[A], _ = h4[A], w = v.tagName.toLowerCase(), E = `${_}${w}`, $ = `${_}*`;
        if (!z.has(w) || !g.has(E) || A !== C && w !== x)
          k.removeChild(v);
        else {
          const M = v.getAttributeNames(), B = M.length;
          if (B) {
            for (let G = 0; G < B; G++) {
              const U = M[G], j = p[U];
              (!j || !j.has($) && !j.has(E)) && v.removeAttribute(U);
            }
            if (Vh(v))
              if (Xh(v)) {
                const G = v.getAttribute("href");
                G && Zh(G) && Tr(v.protocol) && v.removeAttribute("href");
              } else Wh(v) ? Tr(v.action) && v.removeAttribute("action") : Yh(v) ? Tr(v.formAction) && v.removeAttribute("formaction") : jh(v) && (Tr(v.src) && v.removeAttribute("formaction"), v.setAttribute("sandbox", "allow-scripts"));
          }
        }
      }), d;
    }, this.sanitizeFor = (d, g) => {
      throw new Error('"sanitizeFor" is not implemented yet');
    };
    const { allowComments: t, allowCustomElements: r, allowUnknownMarkup: a, blockElements: i, dropElements: l, dropAttributes: s } = e;
    if (t === !1)
      throw new Error('A false "allowComments" is not supported yet');
    if (r)
      throw new Error('A true "allowCustomElements" is not supported yet');
    if (a)
      throw new Error('A true "allowUnknownMarkup" is not supported yet');
    if (i)
      throw new Error('"blockElements" is not supported yet, use "allowElements" instead');
    if (l)
      throw new Error('"dropElements" is not supported yet, use "allowElements" instead');
    if (s)
      throw new Error('"dropAttributes" is not supported yet, use "allowAttributes" instead');
    Pn(this, u0, da(Yo), "f");
    const { allowElements: u, allowAttributes: h } = e;
    u && (_0(this, u0, "f").allowElements = e.allowElements), h && (_0(this, u0, "f").allowAttributes = e.allowAttributes), Pn(this, Ur, new Set(_0(this, u0, "f").allowElements), "f"), Pn(this, Gr, Object.fromEntries(Object.entries(_0(this, u0, "f").allowAttributes || {}).map(([d, g]) => [d, new Set(g)])), "f");
  }
}
u0 = /* @__PURE__ */ new WeakMap(), Ur = /* @__PURE__ */ new WeakMap(), Gr = /* @__PURE__ */ new WeakMap();
Xo.getDefaultConfiguration = () => da(Yo);
const m4 = (n, e) => {
  try {
    return !!n && new URL(n).origin !== new URL(e).origin;
  } catch {
    return !1;
  }
};
function vl(n, e) {
  const t = new Xo(), r = new DOMParser().parseFromString(n, "text/html");
  return Zo(r.body, "A", (a) => {
    a instanceof HTMLElement && "target" in a && m4(a.getAttribute("href"), e) && (a.setAttribute("target", "_blank"), a.setAttribute("rel", "noopener noreferrer"));
  }), t.sanitize(r).body.innerHTML;
}
function Zo(n, e, t) {
  n && (n.nodeName === e || typeof e == "function") && t(n);
  const r = (n == null ? void 0 : n.childNodes) || [];
  for (let a = 0; a < r.length; a++)
    Zo(r[a], e, t);
}
const _l = [
  "!--",
  "!doctype",
  "a",
  "abbr",
  "acronym",
  "address",
  "applet",
  "area",
  "article",
  "aside",
  "audio",
  "b",
  "base",
  "basefont",
  "bdi",
  "bdo",
  "big",
  "blockquote",
  "body",
  "br",
  "button",
  "canvas",
  "caption",
  "center",
  "cite",
  "code",
  "col",
  "colgroup",
  "data",
  "datalist",
  "dd",
  "del",
  "details",
  "dfn",
  "dialog",
  "dir",
  "div",
  "dl",
  "dt",
  "em",
  "embed",
  "fieldset",
  "figcaption",
  "figure",
  "font",
  "footer",
  "form",
  "frame",
  "frameset",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "head",
  "header",
  "hgroup",
  "hr",
  "html",
  "i",
  "iframe",
  "img",
  "input",
  "ins",
  "kbd",
  "label",
  "legend",
  "li",
  "link",
  "main",
  "map",
  "mark",
  "menu",
  "meta",
  "meter",
  "nav",
  "noframes",
  "noscript",
  "object",
  "ol",
  "optgroup",
  "option",
  "output",
  "p",
  "param",
  "picture",
  "pre",
  "progress",
  "q",
  "rp",
  "rt",
  "ruby",
  "s",
  "samp",
  "script",
  "search",
  "section",
  "select",
  "small",
  "source",
  "span",
  "strike",
  "strong",
  "style",
  "sub",
  "summary",
  "sup",
  "svg",
  "table",
  "tbody",
  "td",
  "template",
  "textarea",
  "tfoot",
  "th",
  "thead",
  "time",
  "title",
  "tr",
  "track",
  "tt",
  "u",
  "ul",
  "var",
  "video",
  "wbr"
], f4 = [
  // Base structural elements
  "g",
  "defs",
  "use",
  "symbol",
  // Shape elements
  "rect",
  "circle",
  "ellipse",
  "line",
  "polyline",
  "polygon",
  "path",
  "image",
  // Text elements
  "text",
  "tspan",
  "textPath",
  // Gradient and effects
  "linearGradient",
  "radialGradient",
  "stop",
  "pattern",
  "clipPath",
  "mask",
  "filter",
  // Filter effects
  "feBlend",
  "feColorMatrix",
  "feComponentTransfer",
  "feComposite",
  "feConvolveMatrix",
  "feDiffuseLighting",
  "feDisplacementMap",
  "feGaussianBlur",
  "feMerge",
  "feMorphology",
  "feOffset",
  "feSpecularLighting",
  "feTurbulence",
  "feMergeNode",
  "feFuncR",
  "feFuncG",
  "feFuncB",
  "feFuncA",
  "feDistantLight",
  "fePointLight",
  "feSpotLight",
  "feFlood",
  "feTile",
  // Animation elements
  "animate",
  "animateTransform",
  "animateMotion",
  "mpath",
  "set",
  // Interactive and other elements
  "view",
  "cursor",
  "foreignObject",
  "desc",
  "title",
  "metadata",
  "switch"
], p4 = [
  ..._l,
  ...f4.filter((n) => !_l.includes(n))
], {
  HtmlTagHydration: g4,
  SvelteComponent: v4,
  attr: _4,
  binding_callbacks: b4,
  children: y4,
  claim_element: w4,
  claim_html_tag: x4,
  detach: bl,
  element: k4,
  init: D4,
  insert_hydration: S4,
  noop: yl,
  safe_not_equal: A4,
  toggle_class: $r
} = window.__gradio__svelte__internal, { afterUpdate: E4, tick: F4, onMount: p2 } = window.__gradio__svelte__internal;
function C4(n) {
  let e, t;
  return {
    c() {
      e = k4("span"), t = new g4(!1), this.h();
    },
    l(r) {
      e = w4(r, "SPAN", { class: !0 });
      var a = y4(e);
      t = x4(a, !1), a.forEach(bl), this.h();
    },
    h() {
      t.a = null, _4(e, "class", "md svelte-1m32c2s"), $r(
        e,
        "chatbot",
        /*chatbot*/
        n[0]
      ), $r(
        e,
        "prose",
        /*render_markdown*/
        n[1]
      );
    },
    m(r, a) {
      S4(r, e, a), t.m(
        /*html*/
        n[3],
        e
      ), n[11](e);
    },
    p(r, [a]) {
      a & /*html*/
      8 && t.p(
        /*html*/
        r[3]
      ), a & /*chatbot*/
      1 && $r(
        e,
        "chatbot",
        /*chatbot*/
        r[0]
      ), a & /*render_markdown*/
      2 && $r(
        e,
        "prose",
        /*render_markdown*/
        r[1]
      );
    },
    i: yl,
    o: yl,
    d(r) {
      r && bl(e), n[11](null);
    }
  };
}
function wl(n) {
  return n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function T4(n, e, t) {
  var r = this && this.__awaiter || function(w, E, T, $) {
    function M(B) {
      return B instanceof T ? B : new T(function(G) {
        G(B);
      });
    }
    return new (T || (T = Promise))(function(B, G) {
      function U(ee) {
        try {
          oe($.next(ee));
        } catch (ue) {
          G(ue);
        }
      }
      function j(ee) {
        try {
          oe($.throw(ee));
        } catch (ue) {
          G(ue);
        }
      }
      function oe(ee) {
        ee.done ? B(ee.value) : M(ee.value).then(U, j);
      }
      oe(($ = $.apply(w, E || [])).next());
    });
  };
  let { chatbot: a = !0 } = e, { message: i } = e, { sanitize_html: l = !0 } = e, { latex_delimiters: s = [] } = e, { render_markdown: u = !0 } = e, { line_breaks: h = !0 } = e, { header_links: d = !1 } = e, { allow_tags: g = !1 } = e, { theme_mode: p = "system" } = e, v, k;
  const A = Uh({
    header_links: d,
    line_breaks: h,
    latex_delimiters: s || []
  });
  function C(w, E) {
    if (E === !0) {
      const T = /<\/?([a-zA-Z][a-zA-Z0-9-]*)([\s>])/g;
      return w.replace(T, ($, M, B) => p4.includes(M.toLowerCase()) ? $ : $.replace(/</g, "&lt;").replace(/>/g, "&gt;"));
    }
    if (Array.isArray(E)) {
      const T = E.map((M) => ({
        open: new RegExp(`<(${M})(\\s+[^>]*)?>`, "gi"),
        close: new RegExp(`</(${M})>`, "gi")
      }));
      let $ = w;
      return T.forEach((M) => {
        $ = $.replace(M.open, (B) => B.replace(/</g, "&lt;").replace(/>/g, "&gt;")), $ = $.replace(M.close, (B) => B.replace(/</g, "&lt;").replace(/>/g, "&gt;"));
      }), $;
    }
    return w;
  }
  function z(w) {
    let E = w;
    if (u) {
      const T = [];
      s.forEach(($, M) => {
        const B = wl($.left), G = wl($.right), U = new RegExp(`${B}([\\s\\S]+?)${G}`, "g");
        E = E.replace(U, (j, oe) => (T.push(j), `%%%LATEX_BLOCK_${T.length - 1}%%%`));
      }), E = A.parse(E), E = E.replace(/%%%LATEX_BLOCK_(\d+)%%%/g, ($, M) => T[parseInt(M, 10)]);
    }
    return g && (E = C(E, g)), l && vl && (E = vl(E)), E;
  }
  function x(w) {
    return r(this, void 0, void 0, function* () {
      if (s.length > 0 && w && s.some((T) => w.includes(T.left) && w.includes(T.right)) && Uc(v, {
        delimiters: s,
        throwOnError: !1
      }), v) {
        const E = v.querySelectorAll(".mermaid");
        if (E.length > 0) {
          yield F4();
          const { default: T } = yield import("./mermaid.core-VNpJvtL_.js").then(($) => $.bB);
          T.initialize({
            startOnLoad: !1,
            theme: p === "dark" ? "dark" : "default",
            securityLevel: "antiscript"
          }), yield T.run({
            nodes: Array.from(E).map(($) => $)
          });
        }
      }
    });
  }
  E4(() => r(void 0, void 0, void 0, function* () {
    v && document.body.contains(v) ? yield x(i) : console.error("Element is not in the DOM");
  }));
  function _(w) {
    b4[w ? "unshift" : "push"](() => {
      v = w, t(2, v);
    });
  }
  return n.$$set = (w) => {
    "chatbot" in w && t(0, a = w.chatbot), "message" in w && t(4, i = w.message), "sanitize_html" in w && t(5, l = w.sanitize_html), "latex_delimiters" in w && t(6, s = w.latex_delimiters), "render_markdown" in w && t(1, u = w.render_markdown), "line_breaks" in w && t(7, h = w.line_breaks), "header_links" in w && t(8, d = w.header_links), "allow_tags" in w && t(9, g = w.allow_tags), "theme_mode" in w && t(10, p = w.theme_mode);
  }, n.$$.update = () => {
    n.$$.dirty & /*message*/
    16 && (i && i.trim() ? t(3, k = z(i)) : t(3, k = ""));
  }, [
    a,
    u,
    v,
    k,
    i,
    l,
    s,
    h,
    d,
    g,
    p,
    _
  ];
}
class $4 extends v4 {
  constructor(e) {
    super(), D4(this, e, T4, C4, A4, {
      chatbot: 0,
      message: 4,
      sanitize_html: 5,
      latex_delimiters: 6,
      render_markdown: 1,
      line_breaks: 7,
      header_links: 8,
      allow_tags: 9,
      theme_mode: 10
    });
  }
}
const {
  SvelteComponent: M4,
  attr: z4,
  children: B4,
  claim_component: R4,
  claim_element: N4,
  create_component: q4,
  destroy_component: L4,
  detach: xl,
  element: I4,
  init: O4,
  insert_hydration: P4,
  mount_component: H4,
  safe_not_equal: U4,
  transition_in: G4,
  transition_out: V4
} = window.__gradio__svelte__internal;
function W4(n) {
  let e, t, r;
  return t = new $4({
    props: {
      message: (
        /*info*/
        n[0]
      ),
      sanitize_html: !0
    }
  }), {
    c() {
      e = I4("div"), q4(t.$$.fragment), this.h();
    },
    l(a) {
      e = N4(a, "DIV", { class: !0 });
      var i = B4(e);
      R4(t.$$.fragment, i), i.forEach(xl), this.h();
    },
    h() {
      z4(e, "class", "svelte-17qq50w");
    },
    m(a, i) {
      P4(a, e, i), H4(t, e, null), r = !0;
    },
    p(a, [i]) {
      const l = {};
      i & /*info*/
      1 && (l.message = /*info*/
      a[0]), t.$set(l);
    },
    i(a) {
      r || (G4(t.$$.fragment, a), r = !0);
    },
    o(a) {
      V4(t.$$.fragment, a), r = !1;
    },
    d(a) {
      a && xl(e), L4(t);
    }
  };
}
function j4(n, e, t) {
  let { info: r } = e;
  return n.$$set = (a) => {
    "info" in a && t(0, r = a.info);
  }, [r];
}
class Y4 extends M4 {
  constructor(e) {
    super(), O4(this, e, j4, W4, U4, { info: 0 });
  }
}
const {
  SvelteComponent: X4,
  attr: Mr,
  check_outros: Z4,
  children: K4,
  claim_component: Q4,
  claim_element: J4,
  claim_space: ed,
  create_component: td,
  create_slot: rd,
  destroy_component: nd,
  detach: zr,
  element: ad,
  empty: kl,
  get_all_dirty_from_scope: id,
  get_slot_changes: ld,
  group_outros: sd,
  init: od,
  insert_hydration: Hn,
  mount_component: ud,
  safe_not_equal: cd,
  space: hd,
  toggle_class: M0,
  transition_in: Q0,
  transition_out: Vr,
  update_slot_base: dd
} = window.__gradio__svelte__internal;
function Dl(n) {
  let e, t;
  return e = new Y4({ props: { info: (
    /*info*/
    n[1]
  ) } }), {
    c() {
      td(e.$$.fragment);
    },
    l(r) {
      Q4(e.$$.fragment, r);
    },
    m(r, a) {
      ud(e, r, a), t = !0;
    },
    p(r, a) {
      const i = {};
      a & /*info*/
      2 && (i.info = /*info*/
      r[1]), e.$set(i);
    },
    i(r) {
      t || (Q0(e.$$.fragment, r), t = !0);
    },
    o(r) {
      Vr(e.$$.fragment, r), t = !1;
    },
    d(r) {
      nd(e, r);
    }
  };
}
function md(n) {
  let e, t, r, a, i;
  const l = (
    /*#slots*/
    n[4].default
  ), s = rd(
    l,
    n,
    /*$$scope*/
    n[3],
    null
  );
  let u = (
    /*info*/
    n[1] && Dl(n)
  );
  return {
    c() {
      e = ad("span"), s && s.c(), r = hd(), u && u.c(), a = kl(), this.h();
    },
    l(h) {
      e = J4(h, "SPAN", {
        "data-testid": !0,
        dir: !0,
        class: !0
      });
      var d = K4(e);
      s && s.l(d), d.forEach(zr), r = ed(h), u && u.l(h), a = kl(), this.h();
    },
    h() {
      Mr(e, "data-testid", "block-info"), Mr(e, "dir", t = /*rtl*/
      n[2] ? "rtl" : "ltr"), Mr(e, "class", "svelte-zgrq3"), M0(e, "sr-only", !/*show_label*/
      n[0]), M0(e, "hide", !/*show_label*/
      n[0]), M0(
        e,
        "has-info",
        /*info*/
        n[1] != null
      );
    },
    m(h, d) {
      Hn(h, e, d), s && s.m(e, null), Hn(h, r, d), u && u.m(h, d), Hn(h, a, d), i = !0;
    },
    p(h, [d]) {
      s && s.p && (!i || d & /*$$scope*/
      8) && dd(
        s,
        l,
        h,
        /*$$scope*/
        h[3],
        i ? ld(
          l,
          /*$$scope*/
          h[3],
          d,
          null
        ) : id(
          /*$$scope*/
          h[3]
        ),
        null
      ), (!i || d & /*rtl*/
      4 && t !== (t = /*rtl*/
      h[2] ? "rtl" : "ltr")) && Mr(e, "dir", t), (!i || d & /*show_label*/
      1) && M0(e, "sr-only", !/*show_label*/
      h[0]), (!i || d & /*show_label*/
      1) && M0(e, "hide", !/*show_label*/
      h[0]), (!i || d & /*info*/
      2) && M0(
        e,
        "has-info",
        /*info*/
        h[1] != null
      ), /*info*/
      h[1] ? u ? (u.p(h, d), d & /*info*/
      2 && Q0(u, 1)) : (u = Dl(h), u.c(), Q0(u, 1), u.m(a.parentNode, a)) : u && (sd(), Vr(u, 1, 1, () => {
        u = null;
      }), Z4());
    },
    i(h) {
      i || (Q0(s, h), Q0(u), i = !0);
    },
    o(h) {
      Vr(s, h), Vr(u), i = !1;
    },
    d(h) {
      h && (zr(e), zr(r), zr(a)), s && s.d(h), u && u.d(h);
    }
  };
}
function fd(n, e, t) {
  let { $$slots: r = {}, $$scope: a } = e, { show_label: i = !0 } = e, { info: l = void 0 } = e, { rtl: s = !1 } = e;
  return n.$$set = (u) => {
    "show_label" in u && t(0, i = u.show_label), "info" in u && t(1, l = u.info), "rtl" in u && t(2, s = u.rtl), "$$scope" in u && t(3, a = u.$$scope);
  }, [i, l, s, a, r];
}
class pd extends X4 {
  constructor(e) {
    super(), od(this, e, fd, md, cd, { show_label: 0, info: 1, rtl: 2 });
  }
}
const {
  SvelteComponent: g2,
  append_hydration: v2,
  attr: _2,
  children: b2,
  claim_component: y2,
  claim_element: w2,
  claim_space: x2,
  claim_text: k2,
  create_component: D2,
  destroy_component: S2,
  detach: A2,
  element: E2,
  init: F2,
  insert_hydration: C2,
  mount_component: T2,
  safe_not_equal: $2,
  set_data: M2,
  space: z2,
  text: B2,
  toggle_class: R2,
  transition_in: N2,
  transition_out: q2
} = window.__gradio__svelte__internal, {
  SvelteComponent: gd,
  append_hydration: Wr,
  attr: Kt,
  bubble: vd,
  check_outros: _d,
  children: ma,
  claim_component: bd,
  claim_element: fa,
  claim_space: Sl,
  claim_text: yd,
  construct_svelte_component: Al,
  create_component: El,
  create_slot: wd,
  destroy_component: Fl,
  detach: ar,
  element: pa,
  get_all_dirty_from_scope: xd,
  get_slot_changes: kd,
  group_outros: Dd,
  init: Sd,
  insert_hydration: Ko,
  listen: Ad,
  mount_component: Cl,
  safe_not_equal: Ed,
  set_data: Fd,
  set_style: Br,
  space: Tl,
  text: Cd,
  toggle_class: qe,
  transition_in: Un,
  transition_out: Gn,
  update_slot_base: Td
} = window.__gradio__svelte__internal;
function $l(n) {
  let e, t;
  return {
    c() {
      e = pa("span"), t = Cd(
        /*label*/
        n[1]
      ), this.h();
    },
    l(r) {
      e = fa(r, "SPAN", { class: !0 });
      var a = ma(e);
      t = yd(
        a,
        /*label*/
        n[1]
      ), a.forEach(ar), this.h();
    },
    h() {
      Kt(e, "class", "svelte-qgco6m");
    },
    m(r, a) {
      Ko(r, e, a), Wr(e, t);
    },
    p(r, a) {
      a & /*label*/
      2 && Fd(
        t,
        /*label*/
        r[1]
      );
    },
    d(r) {
      r && ar(e);
    }
  };
}
function $d(n) {
  let e, t, r, a, i, l, s, u, h = (
    /*show_label*/
    n[2] && $l(n)
  );
  var d = (
    /*Icon*/
    n[0]
  );
  function g(k, A) {
    return {};
  }
  d && (a = Al(d, g()));
  const p = (
    /*#slots*/
    n[14].default
  ), v = wd(
    p,
    n,
    /*$$scope*/
    n[13],
    null
  );
  return {
    c() {
      e = pa("button"), h && h.c(), t = Tl(), r = pa("div"), a && El(a.$$.fragment), i = Tl(), v && v.c(), this.h();
    },
    l(k) {
      e = fa(k, "BUTTON", {
        "aria-label": !0,
        "aria-haspopup": !0,
        title: !0,
        class: !0
      });
      var A = ma(e);
      h && h.l(A), t = Sl(A), r = fa(A, "DIV", { class: !0 });
      var C = ma(r);
      a && bd(a.$$.fragment, C), i = Sl(C), v && v.l(C), C.forEach(ar), A.forEach(ar), this.h();
    },
    h() {
      Kt(r, "class", "svelte-qgco6m"), qe(
        r,
        "x-small",
        /*size*/
        n[4] === "x-small"
      ), qe(
        r,
        "small",
        /*size*/
        n[4] === "small"
      ), qe(
        r,
        "large",
        /*size*/
        n[4] === "large"
      ), qe(
        r,
        "medium",
        /*size*/
        n[4] === "medium"
      ), e.disabled = /*disabled*/
      n[7], Kt(
        e,
        "aria-label",
        /*label*/
        n[1]
      ), Kt(
        e,
        "aria-haspopup",
        /*hasPopup*/
        n[8]
      ), Kt(
        e,
        "title",
        /*label*/
        n[1]
      ), Kt(e, "class", "svelte-qgco6m"), qe(
        e,
        "pending",
        /*pending*/
        n[3]
      ), qe(
        e,
        "padded",
        /*padded*/
        n[5]
      ), qe(
        e,
        "highlight",
        /*highlight*/
        n[6]
      ), qe(
        e,
        "transparent",
        /*transparent*/
        n[9]
      ), Br(e, "color", !/*disabled*/
      n[7] && /*_color*/
      n[11] ? (
        /*_color*/
        n[11]
      ) : "var(--block-label-text-color)"), Br(e, "--bg-color", /*disabled*/
      n[7] ? "auto" : (
        /*background*/
        n[10]
      ));
    },
    m(k, A) {
      Ko(k, e, A), h && h.m(e, null), Wr(e, t), Wr(e, r), a && Cl(a, r, null), Wr(r, i), v && v.m(r, null), l = !0, s || (u = Ad(
        e,
        "click",
        /*click_handler*/
        n[15]
      ), s = !0);
    },
    p(k, [A]) {
      if (/*show_label*/
      k[2] ? h ? h.p(k, A) : (h = $l(k), h.c(), h.m(e, t)) : h && (h.d(1), h = null), A & /*Icon*/
      1 && d !== (d = /*Icon*/
      k[0])) {
        if (a) {
          Dd();
          const C = a;
          Gn(C.$$.fragment, 1, 0, () => {
            Fl(C, 1);
          }), _d();
        }
        d ? (a = Al(d, g()), El(a.$$.fragment), Un(a.$$.fragment, 1), Cl(a, r, i)) : a = null;
      }
      v && v.p && (!l || A & /*$$scope*/
      8192) && Td(
        v,
        p,
        k,
        /*$$scope*/
        k[13],
        l ? kd(
          p,
          /*$$scope*/
          k[13],
          A,
          null
        ) : xd(
          /*$$scope*/
          k[13]
        ),
        null
      ), (!l || A & /*size*/
      16) && qe(
        r,
        "x-small",
        /*size*/
        k[4] === "x-small"
      ), (!l || A & /*size*/
      16) && qe(
        r,
        "small",
        /*size*/
        k[4] === "small"
      ), (!l || A & /*size*/
      16) && qe(
        r,
        "large",
        /*size*/
        k[4] === "large"
      ), (!l || A & /*size*/
      16) && qe(
        r,
        "medium",
        /*size*/
        k[4] === "medium"
      ), (!l || A & /*disabled*/
      128) && (e.disabled = /*disabled*/
      k[7]), (!l || A & /*label*/
      2) && Kt(
        e,
        "aria-label",
        /*label*/
        k[1]
      ), (!l || A & /*hasPopup*/
      256) && Kt(
        e,
        "aria-haspopup",
        /*hasPopup*/
        k[8]
      ), (!l || A & /*label*/
      2) && Kt(
        e,
        "title",
        /*label*/
        k[1]
      ), (!l || A & /*pending*/
      8) && qe(
        e,
        "pending",
        /*pending*/
        k[3]
      ), (!l || A & /*padded*/
      32) && qe(
        e,
        "padded",
        /*padded*/
        k[5]
      ), (!l || A & /*highlight*/
      64) && qe(
        e,
        "highlight",
        /*highlight*/
        k[6]
      ), (!l || A & /*transparent*/
      512) && qe(
        e,
        "transparent",
        /*transparent*/
        k[9]
      ), A & /*disabled, _color*/
      2176 && Br(e, "color", !/*disabled*/
      k[7] && /*_color*/
      k[11] ? (
        /*_color*/
        k[11]
      ) : "var(--block-label-text-color)"), A & /*disabled, background*/
      1152 && Br(e, "--bg-color", /*disabled*/
      k[7] ? "auto" : (
        /*background*/
        k[10]
      ));
    },
    i(k) {
      l || (a && Un(a.$$.fragment, k), Un(v, k), l = !0);
    },
    o(k) {
      a && Gn(a.$$.fragment, k), Gn(v, k), l = !1;
    },
    d(k) {
      k && ar(e), h && h.d(), a && Fl(a), v && v.d(k), s = !1, u();
    }
  };
}
function Md(n, e, t) {
  let r, { $$slots: a = {}, $$scope: i } = e, { Icon: l } = e, { label: s = "" } = e, { show_label: u = !1 } = e, { pending: h = !1 } = e, { size: d = "small" } = e, { padded: g = !0 } = e, { highlight: p = !1 } = e, { disabled: v = !1 } = e, { hasPopup: k = !1 } = e, { color: A = "var(--block-label-text-color)" } = e, { transparent: C = !1 } = e, { background: z = "var(--block-background-fill)" } = e;
  function x(_) {
    vd.call(this, n, _);
  }
  return n.$$set = (_) => {
    "Icon" in _ && t(0, l = _.Icon), "label" in _ && t(1, s = _.label), "show_label" in _ && t(2, u = _.show_label), "pending" in _ && t(3, h = _.pending), "size" in _ && t(4, d = _.size), "padded" in _ && t(5, g = _.padded), "highlight" in _ && t(6, p = _.highlight), "disabled" in _ && t(7, v = _.disabled), "hasPopup" in _ && t(8, k = _.hasPopup), "color" in _ && t(12, A = _.color), "transparent" in _ && t(9, C = _.transparent), "background" in _ && t(10, z = _.background), "$$scope" in _ && t(13, i = _.$$scope);
  }, n.$$.update = () => {
    n.$$.dirty & /*highlight, color*/
    4160 && t(11, r = p ? "var(--color-accent)" : A);
  }, [
    l,
    s,
    u,
    h,
    d,
    g,
    p,
    v,
    k,
    C,
    z,
    r,
    A,
    i,
    a,
    x
  ];
}
class zd extends gd {
  constructor(e) {
    super(), Sd(this, e, Md, $d, Ed, {
      Icon: 0,
      label: 1,
      show_label: 2,
      pending: 3,
      size: 4,
      padded: 5,
      highlight: 6,
      disabled: 7,
      hasPopup: 8,
      color: 12,
      transparent: 9,
      background: 10
    });
  }
}
const {
  SvelteComponent: L2,
  append_hydration: I2,
  attr: O2,
  binding_callbacks: P2,
  children: H2,
  claim_element: U2,
  create_slot: G2,
  detach: V2,
  element: W2,
  get_all_dirty_from_scope: j2,
  get_slot_changes: Y2,
  init: X2,
  insert_hydration: Z2,
  safe_not_equal: K2,
  toggle_class: Q2,
  transition_in: J2,
  transition_out: ef,
  update_slot_base: tf
} = window.__gradio__svelte__internal, {
  SvelteComponent: rf,
  append_hydration: nf,
  attr: af,
  children: lf,
  claim_svg_element: sf,
  detach: of,
  init: uf,
  insert_hydration: cf,
  noop: hf,
  safe_not_equal: df,
  svg_element: mf
} = window.__gradio__svelte__internal, {
  SvelteComponent: ff,
  append_hydration: pf,
  attr: gf,
  children: vf,
  claim_svg_element: _f,
  detach: bf,
  init: yf,
  insert_hydration: wf,
  noop: xf,
  safe_not_equal: kf,
  svg_element: Df
} = window.__gradio__svelte__internal, {
  SvelteComponent: Sf,
  append_hydration: Af,
  attr: Ef,
  children: Ff,
  claim_svg_element: Cf,
  detach: Tf,
  init: $f,
  insert_hydration: Mf,
  noop: zf,
  safe_not_equal: Bf,
  svg_element: Rf
} = window.__gradio__svelte__internal, {
  SvelteComponent: Nf,
  append_hydration: qf,
  attr: Lf,
  children: If,
  claim_svg_element: Of,
  detach: Pf,
  init: Hf,
  insert_hydration: Uf,
  noop: Gf,
  safe_not_equal: Vf,
  svg_element: Wf
} = window.__gradio__svelte__internal, {
  SvelteComponent: jf,
  append_hydration: Yf,
  attr: Xf,
  children: Zf,
  claim_svg_element: Kf,
  detach: Qf,
  init: Jf,
  insert_hydration: e5,
  noop: t5,
  safe_not_equal: r5,
  svg_element: n5
} = window.__gradio__svelte__internal, {
  SvelteComponent: a5,
  append_hydration: i5,
  attr: l5,
  children: s5,
  claim_svg_element: o5,
  detach: u5,
  init: c5,
  insert_hydration: h5,
  noop: d5,
  safe_not_equal: m5,
  svg_element: f5
} = window.__gradio__svelte__internal, {
  SvelteComponent: p5,
  append_hydration: g5,
  attr: v5,
  children: _5,
  claim_svg_element: b5,
  detach: y5,
  init: w5,
  insert_hydration: x5,
  noop: k5,
  safe_not_equal: D5,
  svg_element: S5
} = window.__gradio__svelte__internal, {
  SvelteComponent: A5,
  append_hydration: E5,
  attr: F5,
  children: C5,
  claim_svg_element: T5,
  detach: $5,
  init: M5,
  insert_hydration: z5,
  noop: B5,
  safe_not_equal: R5,
  svg_element: N5
} = window.__gradio__svelte__internal, {
  SvelteComponent: q5,
  append_hydration: L5,
  attr: I5,
  children: O5,
  claim_svg_element: P5,
  detach: H5,
  init: U5,
  insert_hydration: G5,
  noop: V5,
  safe_not_equal: W5,
  svg_element: j5
} = window.__gradio__svelte__internal, {
  SvelteComponent: Y5,
  append_hydration: X5,
  attr: Z5,
  children: K5,
  claim_svg_element: Q5,
  detach: J5,
  init: e3,
  insert_hydration: t3,
  noop: r3,
  safe_not_equal: n3,
  svg_element: a3
} = window.__gradio__svelte__internal, {
  SvelteComponent: i3,
  append_hydration: l3,
  attr: s3,
  children: o3,
  claim_svg_element: u3,
  detach: c3,
  init: h3,
  insert_hydration: d3,
  noop: m3,
  safe_not_equal: f3,
  svg_element: p3
} = window.__gradio__svelte__internal, {
  SvelteComponent: g3,
  append_hydration: v3,
  attr: _3,
  children: b3,
  claim_svg_element: y3,
  detach: w3,
  init: x3,
  insert_hydration: k3,
  noop: D3,
  safe_not_equal: S3,
  svg_element: A3
} = window.__gradio__svelte__internal, {
  SvelteComponent: Bd,
  append_hydration: Vn,
  attr: vt,
  children: Rr,
  claim_svg_element: Nr,
  detach: V0,
  init: Rd,
  insert_hydration: Nd,
  noop: Wn,
  safe_not_equal: qd,
  set_style: Tt,
  svg_element: qr
} = window.__gradio__svelte__internal;
function Ld(n) {
  let e, t, r, a;
  return {
    c() {
      e = qr("svg"), t = qr("g"), r = qr("path"), a = qr("path"), this.h();
    },
    l(i) {
      e = Nr(i, "svg", {
        width: !0,
        height: !0,
        viewBox: !0,
        version: !0,
        xmlns: !0,
        "xmlns:xlink": !0,
        "xml:space": !0,
        stroke: !0,
        style: !0
      });
      var l = Rr(e);
      t = Nr(l, "g", { transform: !0 });
      var s = Rr(t);
      r = Nr(s, "path", { d: !0, style: !0 }), Rr(r).forEach(V0), s.forEach(V0), a = Nr(l, "path", { d: !0, style: !0 }), Rr(a).forEach(V0), l.forEach(V0), this.h();
    },
    h() {
      vt(r, "d", "M18,6L6.087,17.913"), Tt(r, "fill", "none"), Tt(r, "fill-rule", "nonzero"), Tt(r, "stroke-width", "2px"), vt(t, "transform", "matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"), vt(a, "d", "M4.364,4.364L19.636,19.636"), Tt(a, "fill", "none"), Tt(a, "fill-rule", "nonzero"), Tt(a, "stroke-width", "2px"), vt(e, "width", "100%"), vt(e, "height", "100%"), vt(e, "viewBox", "0 0 24 24"), vt(e, "version", "1.1"), vt(e, "xmlns", "http://www.w3.org/2000/svg"), vt(e, "xmlns:xlink", "http://www.w3.org/1999/xlink"), vt(e, "xml:space", "preserve"), vt(e, "stroke", "currentColor"), Tt(e, "fill-rule", "evenodd"), Tt(e, "clip-rule", "evenodd"), Tt(e, "stroke-linecap", "round"), Tt(e, "stroke-linejoin", "round");
    },
    m(i, l) {
      Nd(i, e, l), Vn(e, t), Vn(t, r), Vn(e, a);
    },
    p: Wn,
    i: Wn,
    o: Wn,
    d(i) {
      i && V0(e);
    }
  };
}
class Id extends Bd {
  constructor(e) {
    super(), Rd(this, e, null, Ld, qd, {});
  }
}
const {
  SvelteComponent: E3,
  append_hydration: F3,
  attr: C3,
  children: T3,
  claim_svg_element: $3,
  detach: M3,
  init: z3,
  insert_hydration: B3,
  noop: R3,
  safe_not_equal: N3,
  svg_element: q3
} = window.__gradio__svelte__internal, {
  SvelteComponent: L3,
  append_hydration: I3,
  attr: O3,
  children: P3,
  claim_svg_element: H3,
  detach: U3,
  init: G3,
  insert_hydration: V3,
  noop: W3,
  safe_not_equal: j3,
  svg_element: Y3
} = window.__gradio__svelte__internal, {
  SvelteComponent: X3,
  append_hydration: Z3,
  attr: K3,
  children: Q3,
  claim_svg_element: J3,
  detach: ep,
  init: tp,
  insert_hydration: rp,
  noop: np,
  safe_not_equal: ap,
  svg_element: ip
} = window.__gradio__svelte__internal, {
  SvelteComponent: lp,
  append_hydration: sp,
  attr: op,
  children: up,
  claim_svg_element: cp,
  detach: hp,
  init: dp,
  insert_hydration: mp,
  noop: fp,
  safe_not_equal: pp,
  svg_element: gp
} = window.__gradio__svelte__internal, {
  SvelteComponent: vp,
  append_hydration: _p,
  attr: bp,
  children: yp,
  claim_svg_element: wp,
  detach: xp,
  init: kp,
  insert_hydration: Dp,
  noop: Sp,
  safe_not_equal: Ap,
  svg_element: Ep
} = window.__gradio__svelte__internal, {
  SvelteComponent: Fp,
  append_hydration: Cp,
  attr: Tp,
  children: $p,
  claim_svg_element: Mp,
  detach: zp,
  init: Bp,
  insert_hydration: Rp,
  noop: Np,
  safe_not_equal: qp,
  svg_element: Lp
} = window.__gradio__svelte__internal, {
  SvelteComponent: Ip,
  append_hydration: Op,
  attr: Pp,
  children: Hp,
  claim_svg_element: Up,
  detach: Gp,
  init: Vp,
  insert_hydration: Wp,
  noop: jp,
  safe_not_equal: Yp,
  svg_element: Xp
} = window.__gradio__svelte__internal, {
  SvelteComponent: Zp,
  append_hydration: Kp,
  attr: Qp,
  children: Jp,
  claim_svg_element: e6,
  detach: t6,
  init: r6,
  insert_hydration: n6,
  noop: a6,
  safe_not_equal: i6,
  svg_element: l6
} = window.__gradio__svelte__internal, {
  SvelteComponent: s6,
  append_hydration: o6,
  attr: u6,
  children: c6,
  claim_svg_element: h6,
  detach: d6,
  init: m6,
  insert_hydration: f6,
  noop: p6,
  safe_not_equal: g6,
  svg_element: v6
} = window.__gradio__svelte__internal, {
  SvelteComponent: _6,
  append_hydration: b6,
  attr: y6,
  children: w6,
  claim_svg_element: x6,
  detach: k6,
  init: D6,
  insert_hydration: S6,
  noop: A6,
  safe_not_equal: E6,
  svg_element: F6
} = window.__gradio__svelte__internal, {
  SvelteComponent: C6,
  append_hydration: T6,
  attr: $6,
  children: M6,
  claim_svg_element: z6,
  detach: B6,
  init: R6,
  insert_hydration: N6,
  noop: q6,
  safe_not_equal: L6,
  svg_element: I6
} = window.__gradio__svelte__internal, {
  SvelteComponent: O6,
  append_hydration: P6,
  attr: H6,
  children: U6,
  claim_svg_element: G6,
  detach: V6,
  init: W6,
  insert_hydration: j6,
  noop: Y6,
  safe_not_equal: X6,
  svg_element: Z6
} = window.__gradio__svelte__internal, {
  SvelteComponent: K6,
  append_hydration: Q6,
  attr: J6,
  children: e7,
  claim_svg_element: t7,
  detach: r7,
  init: n7,
  insert_hydration: a7,
  noop: i7,
  safe_not_equal: l7,
  svg_element: s7
} = window.__gradio__svelte__internal, {
  SvelteComponent: o7,
  append_hydration: u7,
  attr: c7,
  children: h7,
  claim_svg_element: d7,
  detach: m7,
  init: f7,
  insert_hydration: p7,
  noop: g7,
  safe_not_equal: v7,
  svg_element: _7
} = window.__gradio__svelte__internal, {
  SvelteComponent: b7,
  append_hydration: y7,
  attr: w7,
  children: x7,
  claim_svg_element: k7,
  detach: D7,
  init: S7,
  insert_hydration: A7,
  noop: E7,
  safe_not_equal: F7,
  svg_element: C7
} = window.__gradio__svelte__internal, {
  SvelteComponent: T7,
  append_hydration: $7,
  attr: M7,
  children: z7,
  claim_svg_element: B7,
  detach: R7,
  init: N7,
  insert_hydration: q7,
  noop: L7,
  safe_not_equal: I7,
  svg_element: O7
} = window.__gradio__svelte__internal, {
  SvelteComponent: P7,
  append_hydration: H7,
  attr: U7,
  children: G7,
  claim_svg_element: V7,
  detach: W7,
  init: j7,
  insert_hydration: Y7,
  noop: X7,
  safe_not_equal: Z7,
  svg_element: K7
} = window.__gradio__svelte__internal, {
  SvelteComponent: Q7,
  append_hydration: J7,
  attr: e8,
  children: t8,
  claim_svg_element: r8,
  detach: n8,
  init: a8,
  insert_hydration: i8,
  noop: l8,
  safe_not_equal: s8,
  svg_element: o8
} = window.__gradio__svelte__internal, {
  SvelteComponent: u8,
  append_hydration: c8,
  attr: h8,
  children: d8,
  claim_svg_element: m8,
  detach: f8,
  init: p8,
  insert_hydration: g8,
  noop: v8,
  safe_not_equal: _8,
  svg_element: b8
} = window.__gradio__svelte__internal, {
  SvelteComponent: y8,
  append_hydration: w8,
  attr: x8,
  children: k8,
  claim_svg_element: D8,
  detach: S8,
  init: A8,
  insert_hydration: E8,
  noop: F8,
  safe_not_equal: C8,
  svg_element: T8
} = window.__gradio__svelte__internal, {
  SvelteComponent: $8,
  append_hydration: M8,
  attr: z8,
  children: B8,
  claim_svg_element: R8,
  detach: N8,
  init: q8,
  insert_hydration: L8,
  noop: I8,
  safe_not_equal: O8,
  svg_element: P8
} = window.__gradio__svelte__internal, {
  SvelteComponent: H8,
  append_hydration: U8,
  attr: G8,
  children: V8,
  claim_svg_element: W8,
  detach: j8,
  init: Y8,
  insert_hydration: X8,
  noop: Z8,
  safe_not_equal: K8,
  svg_element: Q8
} = window.__gradio__svelte__internal, {
  SvelteComponent: J8,
  append_hydration: e9,
  attr: t9,
  children: r9,
  claim_svg_element: n9,
  detach: a9,
  init: i9,
  insert_hydration: l9,
  noop: s9,
  safe_not_equal: o9,
  svg_element: u9
} = window.__gradio__svelte__internal, {
  SvelteComponent: c9,
  append_hydration: h9,
  attr: d9,
  children: m9,
  claim_svg_element: f9,
  detach: p9,
  init: g9,
  insert_hydration: v9,
  noop: _9,
  safe_not_equal: b9,
  svg_element: y9
} = window.__gradio__svelte__internal, {
  SvelteComponent: w9,
  append_hydration: x9,
  attr: k9,
  children: D9,
  claim_svg_element: S9,
  detach: A9,
  init: E9,
  insert_hydration: F9,
  noop: C9,
  safe_not_equal: T9,
  svg_element: $9
} = window.__gradio__svelte__internal, {
  SvelteComponent: M9,
  append_hydration: z9,
  attr: B9,
  children: R9,
  claim_svg_element: N9,
  detach: q9,
  init: L9,
  insert_hydration: I9,
  noop: O9,
  safe_not_equal: P9,
  svg_element: H9
} = window.__gradio__svelte__internal, {
  SvelteComponent: U9,
  append_hydration: G9,
  attr: V9,
  children: W9,
  claim_svg_element: j9,
  detach: Y9,
  init: X9,
  insert_hydration: Z9,
  noop: K9,
  safe_not_equal: Q9,
  svg_element: J9
} = window.__gradio__svelte__internal, {
  SvelteComponent: eg,
  append_hydration: tg,
  attr: rg,
  children: ng,
  claim_svg_element: ag,
  detach: ig,
  init: lg,
  insert_hydration: sg,
  noop: og,
  safe_not_equal: ug,
  svg_element: cg
} = window.__gradio__svelte__internal, {
  SvelteComponent: hg,
  append_hydration: dg,
  attr: mg,
  children: fg,
  claim_svg_element: pg,
  detach: gg,
  init: vg,
  insert_hydration: _g,
  noop: bg,
  safe_not_equal: yg,
  svg_element: wg
} = window.__gradio__svelte__internal, {
  SvelteComponent: xg,
  append_hydration: kg,
  attr: Dg,
  children: Sg,
  claim_svg_element: Ag,
  detach: Eg,
  init: Fg,
  insert_hydration: Cg,
  noop: Tg,
  safe_not_equal: $g,
  svg_element: Mg
} = window.__gradio__svelte__internal, {
  SvelteComponent: zg,
  append_hydration: Bg,
  attr: Rg,
  children: Ng,
  claim_svg_element: qg,
  detach: Lg,
  init: Ig,
  insert_hydration: Og,
  noop: Pg,
  safe_not_equal: Hg,
  svg_element: Ug
} = window.__gradio__svelte__internal, {
  SvelteComponent: Gg,
  append_hydration: Vg,
  attr: Wg,
  children: jg,
  claim_svg_element: Yg,
  detach: Xg,
  init: Zg,
  insert_hydration: Kg,
  noop: Qg,
  safe_not_equal: Jg,
  svg_element: ev
} = window.__gradio__svelte__internal, {
  SvelteComponent: tv,
  append_hydration: rv,
  attr: nv,
  children: av,
  claim_svg_element: iv,
  detach: lv,
  init: sv,
  insert_hydration: ov,
  noop: uv,
  safe_not_equal: cv,
  svg_element: hv
} = window.__gradio__svelte__internal, {
  SvelteComponent: dv,
  append_hydration: mv,
  attr: fv,
  children: pv,
  claim_svg_element: gv,
  detach: vv,
  init: _v,
  insert_hydration: bv,
  noop: yv,
  safe_not_equal: wv,
  svg_element: xv
} = window.__gradio__svelte__internal, {
  SvelteComponent: kv,
  append_hydration: Dv,
  attr: Sv,
  children: Av,
  claim_svg_element: Ev,
  detach: Fv,
  init: Cv,
  insert_hydration: Tv,
  noop: $v,
  safe_not_equal: Mv,
  set_style: zv,
  svg_element: Bv
} = window.__gradio__svelte__internal, {
  SvelteComponent: Rv,
  append_hydration: Nv,
  attr: qv,
  children: Lv,
  claim_svg_element: Iv,
  detach: Ov,
  init: Pv,
  insert_hydration: Hv,
  noop: Uv,
  safe_not_equal: Gv,
  svg_element: Vv
} = window.__gradio__svelte__internal, {
  SvelteComponent: Wv,
  append_hydration: jv,
  attr: Yv,
  children: Xv,
  claim_svg_element: Zv,
  detach: Kv,
  init: Qv,
  insert_hydration: Jv,
  noop: e_,
  safe_not_equal: t_,
  svg_element: r_
} = window.__gradio__svelte__internal, {
  SvelteComponent: n_,
  append_hydration: a_,
  attr: i_,
  children: l_,
  claim_svg_element: s_,
  detach: o_,
  init: u_,
  insert_hydration: c_,
  noop: h_,
  safe_not_equal: d_,
  svg_element: m_
} = window.__gradio__svelte__internal, {
  SvelteComponent: f_,
  append_hydration: p_,
  attr: g_,
  children: v_,
  claim_svg_element: __,
  detach: b_,
  init: y_,
  insert_hydration: w_,
  noop: x_,
  safe_not_equal: k_,
  svg_element: D_
} = window.__gradio__svelte__internal, {
  SvelteComponent: S_,
  append_hydration: A_,
  attr: E_,
  children: F_,
  claim_svg_element: C_,
  detach: T_,
  init: $_,
  insert_hydration: M_,
  noop: z_,
  safe_not_equal: B_,
  svg_element: R_
} = window.__gradio__svelte__internal, {
  SvelteComponent: N_,
  append_hydration: q_,
  attr: L_,
  children: I_,
  claim_svg_element: O_,
  detach: P_,
  init: H_,
  insert_hydration: U_,
  noop: G_,
  safe_not_equal: V_,
  svg_element: W_
} = window.__gradio__svelte__internal, {
  SvelteComponent: j_,
  append_hydration: Y_,
  attr: X_,
  children: Z_,
  claim_svg_element: K_,
  detach: Q_,
  init: J_,
  insert_hydration: eb,
  noop: tb,
  safe_not_equal: rb,
  svg_element: nb
} = window.__gradio__svelte__internal, {
  SvelteComponent: ab,
  append_hydration: ib,
  attr: lb,
  children: sb,
  claim_svg_element: ob,
  detach: ub,
  init: cb,
  insert_hydration: hb,
  noop: db,
  safe_not_equal: mb,
  svg_element: fb
} = window.__gradio__svelte__internal, {
  SvelteComponent: pb,
  append_hydration: gb,
  attr: vb,
  children: _b,
  claim_svg_element: bb,
  claim_text: yb,
  detach: wb,
  init: xb,
  insert_hydration: kb,
  noop: Db,
  safe_not_equal: Sb,
  svg_element: Ab,
  text: Eb
} = window.__gradio__svelte__internal, {
  SvelteComponent: Fb,
  append_hydration: Cb,
  attr: Tb,
  children: $b,
  claim_svg_element: Mb,
  detach: zb,
  init: Bb,
  insert_hydration: Rb,
  noop: Nb,
  safe_not_equal: qb,
  svg_element: Lb
} = window.__gradio__svelte__internal, {
  SvelteComponent: Ib,
  append_hydration: Ob,
  attr: Pb,
  children: Hb,
  claim_svg_element: Ub,
  detach: Gb,
  init: Vb,
  insert_hydration: Wb,
  noop: jb,
  safe_not_equal: Yb,
  svg_element: Xb
} = window.__gradio__svelte__internal, {
  SvelteComponent: Zb,
  append_hydration: Kb,
  attr: Qb,
  children: Jb,
  claim_svg_element: ey,
  detach: ty,
  init: ry,
  insert_hydration: ny,
  noop: ay,
  safe_not_equal: iy,
  svg_element: ly
} = window.__gradio__svelte__internal, {
  SvelteComponent: sy,
  append_hydration: oy,
  attr: uy,
  children: cy,
  claim_svg_element: hy,
  detach: dy,
  init: my,
  insert_hydration: fy,
  noop: py,
  safe_not_equal: gy,
  svg_element: vy
} = window.__gradio__svelte__internal, {
  SvelteComponent: _y,
  append_hydration: by,
  attr: yy,
  children: wy,
  claim_svg_element: xy,
  detach: ky,
  init: Dy,
  insert_hydration: Sy,
  noop: Ay,
  safe_not_equal: Ey,
  svg_element: Fy
} = window.__gradio__svelte__internal, {
  SvelteComponent: Cy,
  append_hydration: Ty,
  attr: $y,
  children: My,
  claim_svg_element: zy,
  detach: By,
  init: Ry,
  insert_hydration: Ny,
  noop: qy,
  safe_not_equal: Ly,
  svg_element: Iy
} = window.__gradio__svelte__internal, {
  SvelteComponent: Oy,
  append_hydration: Py,
  attr: Hy,
  children: Uy,
  claim_svg_element: Gy,
  detach: Vy,
  init: Wy,
  insert_hydration: jy,
  noop: Yy,
  safe_not_equal: Xy,
  svg_element: Zy
} = window.__gradio__svelte__internal, {
  SvelteComponent: Ky,
  append_hydration: Qy,
  attr: Jy,
  children: ew,
  claim_svg_element: tw,
  claim_text: rw,
  detach: nw,
  init: aw,
  insert_hydration: iw,
  noop: lw,
  safe_not_equal: sw,
  svg_element: ow,
  text: uw
} = window.__gradio__svelte__internal, {
  SvelteComponent: cw,
  append_hydration: hw,
  attr: dw,
  children: mw,
  claim_svg_element: fw,
  claim_text: pw,
  detach: gw,
  init: vw,
  insert_hydration: _w,
  noop: bw,
  safe_not_equal: yw,
  svg_element: ww,
  text: xw
} = window.__gradio__svelte__internal, {
  SvelteComponent: kw,
  append_hydration: Dw,
  attr: Sw,
  children: Aw,
  claim_svg_element: Ew,
  claim_text: Fw,
  detach: Cw,
  init: Tw,
  insert_hydration: $w,
  noop: Mw,
  safe_not_equal: zw,
  svg_element: Bw,
  text: Rw
} = window.__gradio__svelte__internal, {
  SvelteComponent: Nw,
  append_hydration: qw,
  attr: Lw,
  children: Iw,
  claim_svg_element: Ow,
  detach: Pw,
  init: Hw,
  insert_hydration: Uw,
  noop: Gw,
  safe_not_equal: Vw,
  svg_element: Ww
} = window.__gradio__svelte__internal, {
  SvelteComponent: jw,
  append_hydration: Yw,
  attr: Xw,
  children: Zw,
  claim_svg_element: Kw,
  detach: Qw,
  init: Jw,
  insert_hydration: ex,
  noop: tx,
  safe_not_equal: rx,
  svg_element: nx
} = window.__gradio__svelte__internal, {
  SvelteComponent: ax,
  append_hydration: ix,
  attr: lx,
  children: sx,
  claim_svg_element: ox,
  detach: ux,
  init: cx,
  insert_hydration: hx,
  noop: dx,
  safe_not_equal: mx,
  svg_element: fx
} = window.__gradio__svelte__internal, {
  SvelteComponent: px,
  append_hydration: gx,
  attr: vx,
  children: _x,
  claim_svg_element: bx,
  detach: yx,
  init: wx,
  insert_hydration: xx,
  noop: kx,
  safe_not_equal: Dx,
  svg_element: Sx
} = window.__gradio__svelte__internal, {
  SvelteComponent: Ax,
  append_hydration: Ex,
  attr: Fx,
  children: Cx,
  claim_svg_element: Tx,
  detach: $x,
  init: Mx,
  insert_hydration: zx,
  noop: Bx,
  safe_not_equal: Rx,
  svg_element: Nx
} = window.__gradio__svelte__internal, {
  SvelteComponent: qx,
  append_hydration: Lx,
  attr: Ix,
  children: Ox,
  claim_svg_element: Px,
  detach: Hx,
  init: Ux,
  insert_hydration: Gx,
  noop: Vx,
  safe_not_equal: Wx,
  svg_element: jx
} = window.__gradio__svelte__internal, {
  SvelteComponent: Yx,
  append_hydration: Xx,
  attr: Zx,
  children: Kx,
  claim_svg_element: Qx,
  detach: Jx,
  init: ek,
  insert_hydration: tk,
  noop: rk,
  safe_not_equal: nk,
  svg_element: ak
} = window.__gradio__svelte__internal, {
  SvelteComponent: ik,
  append_hydration: lk,
  attr: sk,
  children: ok,
  claim_svg_element: uk,
  detach: ck,
  init: hk,
  insert_hydration: dk,
  noop: mk,
  safe_not_equal: fk,
  svg_element: pk
} = window.__gradio__svelte__internal, Od = [
  { color: "red", primary: 600, secondary: 100 },
  { color: "green", primary: 600, secondary: 100 },
  { color: "blue", primary: 600, secondary: 100 },
  { color: "yellow", primary: 500, secondary: 100 },
  { color: "purple", primary: 600, secondary: 100 },
  { color: "teal", primary: 600, secondary: 100 },
  { color: "orange", primary: 600, secondary: 100 },
  { color: "cyan", primary: 600, secondary: 100 },
  { color: "lime", primary: 500, secondary: 100 },
  { color: "pink", primary: 600, secondary: 100 }
], Ml = {
  inherit: "inherit",
  current: "currentColor",
  transparent: "transparent",
  black: "#000",
  white: "#fff",
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617"
  },
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
    950: "#030712"
  },
  zinc: {
    50: "#fafafa",
    100: "#f4f4f5",
    200: "#e4e4e7",
    300: "#d4d4d8",
    400: "#a1a1aa",
    500: "#71717a",
    600: "#52525b",
    700: "#3f3f46",
    800: "#27272a",
    900: "#18181b",
    950: "#09090b"
  },
  neutral: {
    50: "#fafafa",
    100: "#f5f5f5",
    200: "#e5e5e5",
    300: "#d4d4d4",
    400: "#a3a3a3",
    500: "#737373",
    600: "#525252",
    700: "#404040",
    800: "#262626",
    900: "#171717",
    950: "#0a0a0a"
  },
  stone: {
    50: "#fafaf9",
    100: "#f5f5f4",
    200: "#e7e5e4",
    300: "#d6d3d1",
    400: "#a8a29e",
    500: "#78716c",
    600: "#57534e",
    700: "#44403c",
    800: "#292524",
    900: "#1c1917",
    950: "#0c0a09"
  },
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
    950: "#450a0a"
  },
  orange: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
    950: "#431407"
  },
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
    950: "#451a03"
  },
  yellow: {
    50: "#fefce8",
    100: "#fef9c3",
    200: "#fef08a",
    300: "#fde047",
    400: "#facc15",
    500: "#eab308",
    600: "#ca8a04",
    700: "#a16207",
    800: "#854d0e",
    900: "#713f12",
    950: "#422006"
  },
  lime: {
    50: "#f7fee7",
    100: "#ecfccb",
    200: "#d9f99d",
    300: "#bef264",
    400: "#a3e635",
    500: "#84cc16",
    600: "#65a30d",
    700: "#4d7c0f",
    800: "#3f6212",
    900: "#365314",
    950: "#1a2e05"
  },
  green: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#14532d",
    950: "#052e16"
  },
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
    950: "#022c22"
  },
  teal: {
    50: "#f0fdfa",
    100: "#ccfbf1",
    200: "#99f6e4",
    300: "#5eead4",
    400: "#2dd4bf",
    500: "#14b8a6",
    600: "#0d9488",
    700: "#0f766e",
    800: "#115e59",
    900: "#134e4a",
    950: "#042f2e"
  },
  cyan: {
    50: "#ecfeff",
    100: "#cffafe",
    200: "#a5f3fc",
    300: "#67e8f9",
    400: "#22d3ee",
    500: "#06b6d4",
    600: "#0891b2",
    700: "#0e7490",
    800: "#155e75",
    900: "#164e63",
    950: "#083344"
  },
  sky: {
    50: "#f0f9ff",
    100: "#e0f2fe",
    200: "#bae6fd",
    300: "#7dd3fc",
    400: "#38bdf8",
    500: "#0ea5e9",
    600: "#0284c7",
    700: "#0369a1",
    800: "#075985",
    900: "#0c4a6e",
    950: "#082f49"
  },
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
    950: "#172554"
  },
  indigo: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b"
  },
  violet: {
    50: "#f5f3ff",
    100: "#ede9fe",
    200: "#ddd6fe",
    300: "#c4b5fd",
    400: "#a78bfa",
    500: "#8b5cf6",
    600: "#7c3aed",
    700: "#6d28d9",
    800: "#5b21b6",
    900: "#4c1d95",
    950: "#2e1065"
  },
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7e22ce",
    800: "#6b21a8",
    900: "#581c87",
    950: "#3b0764"
  },
  fuchsia: {
    50: "#fdf4ff",
    100: "#fae8ff",
    200: "#f5d0fe",
    300: "#f0abfc",
    400: "#e879f9",
    500: "#d946ef",
    600: "#c026d3",
    700: "#a21caf",
    800: "#86198f",
    900: "#701a75",
    950: "#4a044e"
  },
  pink: {
    50: "#fdf2f8",
    100: "#fce7f3",
    200: "#fbcfe8",
    300: "#f9a8d4",
    400: "#f472b6",
    500: "#ec4899",
    600: "#db2777",
    700: "#be185d",
    800: "#9d174d",
    900: "#831843",
    950: "#500724"
  },
  rose: {
    50: "#fff1f2",
    100: "#ffe4e6",
    200: "#fecdd3",
    300: "#fda4af",
    400: "#fb7185",
    500: "#f43f5e",
    600: "#e11d48",
    700: "#be123c",
    800: "#9f1239",
    900: "#881337",
    950: "#4c0519"
  }
};
Od.reduce(
  (n, { color: e, primary: t, secondary: r }) => ({
    ...n,
    [e]: {
      primary: Ml[e][t],
      secondary: Ml[e][r]
    }
  }),
  {}
);
const {
  SvelteComponent: gk,
  claim_component: vk,
  create_component: _k,
  destroy_component: bk,
  init: yk,
  mount_component: wk,
  safe_not_equal: xk,
  transition_in: kk,
  transition_out: Dk
} = window.__gradio__svelte__internal, { createEventDispatcher: Sk } = window.__gradio__svelte__internal, {
  SvelteComponent: Ak,
  append_hydration: Ek,
  attr: Fk,
  check_outros: Ck,
  children: Tk,
  claim_component: $k,
  claim_element: Mk,
  claim_space: zk,
  claim_text: Bk,
  create_component: Rk,
  destroy_component: Nk,
  detach: qk,
  element: Lk,
  empty: Ik,
  group_outros: Ok,
  init: Pk,
  insert_hydration: Hk,
  mount_component: Uk,
  safe_not_equal: Gk,
  set_data: Vk,
  space: Wk,
  text: jk,
  toggle_class: Yk,
  transition_in: Xk,
  transition_out: Zk
} = window.__gradio__svelte__internal, {
  SvelteComponent: Kk,
  attr: Qk,
  children: Jk,
  claim_element: eD,
  create_slot: tD,
  detach: rD,
  element: nD,
  get_all_dirty_from_scope: aD,
  get_slot_changes: iD,
  init: lD,
  insert_hydration: sD,
  safe_not_equal: oD,
  toggle_class: uD,
  transition_in: cD,
  transition_out: hD,
  update_slot_base: dD
} = window.__gradio__svelte__internal, {
  SvelteComponent: mD,
  append_hydration: fD,
  attr: pD,
  check_outros: gD,
  children: vD,
  claim_component: _D,
  claim_element: bD,
  claim_space: yD,
  create_component: wD,
  destroy_component: xD,
  detach: kD,
  element: DD,
  empty: SD,
  group_outros: AD,
  init: ED,
  insert_hydration: FD,
  listen: CD,
  mount_component: TD,
  safe_not_equal: $D,
  space: MD,
  toggle_class: zD,
  transition_in: BD,
  transition_out: RD
} = window.__gradio__svelte__internal, {
  SvelteComponent: ND,
  attr: qD,
  children: LD,
  claim_element: ID,
  create_slot: OD,
  detach: PD,
  element: HD,
  get_all_dirty_from_scope: UD,
  get_slot_changes: GD,
  init: VD,
  insert_hydration: WD,
  null_to_empty: jD,
  safe_not_equal: YD,
  transition_in: XD,
  transition_out: ZD,
  update_slot_base: KD
} = window.__gradio__svelte__internal, {
  SvelteComponent: QD,
  check_outros: JD,
  claim_component: eS,
  create_component: tS,
  destroy_component: rS,
  detach: nS,
  empty: aS,
  group_outros: iS,
  init: lS,
  insert_hydration: sS,
  mount_component: oS,
  noop: uS,
  safe_not_equal: cS,
  transition_in: hS,
  transition_out: dS
} = window.__gradio__svelte__internal, { createEventDispatcher: mS } = window.__gradio__svelte__internal;
function B0(n) {
  let e = ["", "k", "M", "G", "T", "P", "E", "Z"], t = 0;
  for (; n > 1e3 && t < e.length - 1; )
    n /= 1e3, t++;
  let r = e[t];
  return (Number.isInteger(n) ? n : n.toFixed(1)) + r;
}
function jr() {
}
function Pd(n, e) {
  return n != n ? e == e : n !== e || n && typeof n == "object" || typeof n == "function";
}
const Qo = typeof window < "u";
let zl = Qo ? () => window.performance.now() : () => Date.now(), Jo = Qo ? (n) => requestAnimationFrame(n) : jr;
const q0 = /* @__PURE__ */ new Set();
function eu(n) {
  q0.forEach((e) => {
    e.c(n) || (q0.delete(e), e.f());
  }), q0.size !== 0 && Jo(eu);
}
function Hd(n) {
  let e;
  return q0.size === 0 && Jo(eu), {
    promise: new Promise((t) => {
      q0.add(e = { c: n, f: t });
    }),
    abort() {
      q0.delete(e);
    }
  };
}
const z0 = [];
function Ud(n, e = jr) {
  let t;
  const r = /* @__PURE__ */ new Set();
  function a(s) {
    if (Pd(n, s) && (n = s, t)) {
      const u = !z0.length;
      for (const h of r)
        h[1](), z0.push(h, n);
      if (u) {
        for (let h = 0; h < z0.length; h += 2)
          z0[h][0](z0[h + 1]);
        z0.length = 0;
      }
    }
  }
  function i(s) {
    a(s(n));
  }
  function l(s, u = jr) {
    const h = [s, u];
    return r.add(h), r.size === 1 && (t = e(a, i) || jr), s(n), () => {
      r.delete(h), r.size === 0 && t && (t(), t = null);
    };
  }
  return { set: a, update: i, subscribe: l };
}
function Bl(n) {
  return Object.prototype.toString.call(n) === "[object Date]";
}
function ga(n, e, t, r) {
  if (typeof t == "number" || Bl(t)) {
    const a = r - t, i = (t - e) / (n.dt || 1 / 60), l = n.opts.stiffness * a, s = n.opts.damping * i, u = (l - s) * n.inv_mass, h = (i + u) * n.dt;
    return Math.abs(h) < n.opts.precision && Math.abs(a) < n.opts.precision ? r : (n.settled = !1, Bl(t) ? new Date(t.getTime() + h) : t + h);
  } else {
    if (Array.isArray(t))
      return t.map(
        (a, i) => ga(n, e[i], t[i], r[i])
      );
    if (typeof t == "object") {
      const a = {};
      for (const i in t)
        a[i] = ga(n, e[i], t[i], r[i]);
      return a;
    } else
      throw new Error(`Cannot spring ${typeof t} values`);
  }
}
function Rl(n, e = {}) {
  const t = Ud(n), { stiffness: r = 0.15, damping: a = 0.8, precision: i = 0.01 } = e;
  let l, s, u, h = n, d = n, g = 1, p = 0, v = !1;
  function k(C, z = {}) {
    d = C;
    const x = u = {};
    return n == null || z.hard || A.stiffness >= 1 && A.damping >= 1 ? (v = !0, l = zl(), h = C, t.set(n = d), Promise.resolve()) : (z.soft && (p = 1 / ((z.soft === !0 ? 0.5 : +z.soft) * 60), g = 0), s || (l = zl(), v = !1, s = Hd((_) => {
      if (v)
        return v = !1, s = null, !1;
      g = Math.min(g + p, 1);
      const w = {
        inv_mass: g,
        opts: A,
        settled: !0,
        dt: (_ - l) * 60 / 1e3
      }, E = ga(w, h, n, d);
      return l = _, h = n, t.set(n = E), w.settled && (s = null), !w.settled;
    })), new Promise((_) => {
      s.promise.then(() => {
        x === u && _();
      });
    }));
  }
  const A = {
    set: k,
    update: (C, z) => k(C(d, n), z),
    subscribe: t.subscribe,
    stiffness: r,
    damping: a,
    precision: i
  };
  return A;
}
const {
  SvelteComponent: Gd,
  append_hydration: _t,
  attr: ie,
  children: nt,
  claim_element: Vd,
  claim_svg_element: bt,
  component_subscribe: Nl,
  detach: Qe,
  element: Wd,
  init: jd,
  insert_hydration: Yd,
  noop: ql,
  safe_not_equal: Xd,
  set_style: Lr,
  svg_element: yt,
  toggle_class: Ll
} = window.__gradio__svelte__internal, { onMount: Zd } = window.__gradio__svelte__internal;
function Kd(n) {
  let e, t, r, a, i, l, s, u, h, d, g, p;
  return {
    c() {
      e = Wd("div"), t = yt("svg"), r = yt("g"), a = yt("path"), i = yt("path"), l = yt("path"), s = yt("path"), u = yt("g"), h = yt("path"), d = yt("path"), g = yt("path"), p = yt("path"), this.h();
    },
    l(v) {
      e = Vd(v, "DIV", { class: !0 });
      var k = nt(e);
      t = bt(k, "svg", {
        viewBox: !0,
        fill: !0,
        xmlns: !0,
        class: !0
      });
      var A = nt(t);
      r = bt(A, "g", { style: !0 });
      var C = nt(r);
      a = bt(C, "path", {
        d: !0,
        fill: !0,
        "fill-opacity": !0,
        class: !0
      }), nt(a).forEach(Qe), i = bt(C, "path", { d: !0, fill: !0, class: !0 }), nt(i).forEach(Qe), l = bt(C, "path", {
        d: !0,
        fill: !0,
        "fill-opacity": !0,
        class: !0
      }), nt(l).forEach(Qe), s = bt(C, "path", { d: !0, fill: !0, class: !0 }), nt(s).forEach(Qe), C.forEach(Qe), u = bt(A, "g", { style: !0 });
      var z = nt(u);
      h = bt(z, "path", {
        d: !0,
        fill: !0,
        "fill-opacity": !0,
        class: !0
      }), nt(h).forEach(Qe), d = bt(z, "path", { d: !0, fill: !0, class: !0 }), nt(d).forEach(Qe), g = bt(z, "path", {
        d: !0,
        fill: !0,
        "fill-opacity": !0,
        class: !0
      }), nt(g).forEach(Qe), p = bt(z, "path", { d: !0, fill: !0, class: !0 }), nt(p).forEach(Qe), z.forEach(Qe), A.forEach(Qe), k.forEach(Qe), this.h();
    },
    h() {
      ie(a, "d", "M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z"), ie(a, "fill", "#FF7C00"), ie(a, "fill-opacity", "0.4"), ie(a, "class", "svelte-43sxxs"), ie(i, "d", "M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z"), ie(i, "fill", "#FF7C00"), ie(i, "class", "svelte-43sxxs"), ie(l, "d", "M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z"), ie(l, "fill", "#FF7C00"), ie(l, "fill-opacity", "0.4"), ie(l, "class", "svelte-43sxxs"), ie(s, "d", "M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z"), ie(s, "fill", "#FF7C00"), ie(s, "class", "svelte-43sxxs"), Lr(r, "transform", "translate(" + /*$top*/
      n[1][0] + "px, " + /*$top*/
      n[1][1] + "px)"), ie(h, "d", "M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z"), ie(h, "fill", "#FF7C00"), ie(h, "fill-opacity", "0.4"), ie(h, "class", "svelte-43sxxs"), ie(d, "d", "M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z"), ie(d, "fill", "#FF7C00"), ie(d, "class", "svelte-43sxxs"), ie(g, "d", "M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z"), ie(g, "fill", "#FF7C00"), ie(g, "fill-opacity", "0.4"), ie(g, "class", "svelte-43sxxs"), ie(p, "d", "M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z"), ie(p, "fill", "#FF7C00"), ie(p, "class", "svelte-43sxxs"), Lr(u, "transform", "translate(" + /*$bottom*/
      n[2][0] + "px, " + /*$bottom*/
      n[2][1] + "px)"), ie(t, "viewBox", "-1200 -1200 3000 3000"), ie(t, "fill", "none"), ie(t, "xmlns", "http://www.w3.org/2000/svg"), ie(t, "class", "svelte-43sxxs"), ie(e, "class", "svelte-43sxxs"), Ll(
        e,
        "margin",
        /*margin*/
        n[0]
      );
    },
    m(v, k) {
      Yd(v, e, k), _t(e, t), _t(t, r), _t(r, a), _t(r, i), _t(r, l), _t(r, s), _t(t, u), _t(u, h), _t(u, d), _t(u, g), _t(u, p);
    },
    p(v, [k]) {
      k & /*$top*/
      2 && Lr(r, "transform", "translate(" + /*$top*/
      v[1][0] + "px, " + /*$top*/
      v[1][1] + "px)"), k & /*$bottom*/
      4 && Lr(u, "transform", "translate(" + /*$bottom*/
      v[2][0] + "px, " + /*$bottom*/
      v[2][1] + "px)"), k & /*margin*/
      1 && Ll(
        e,
        "margin",
        /*margin*/
        v[0]
      );
    },
    i: ql,
    o: ql,
    d(v) {
      v && Qe(e);
    }
  };
}
function Qd(n, e, t) {
  let r, a;
  var i = this && this.__awaiter || function(v, k, A, C) {
    function z(x) {
      return x instanceof A ? x : new A(function(_) {
        _(x);
      });
    }
    return new (A || (A = Promise))(function(x, _) {
      function w($) {
        try {
          T(C.next($));
        } catch (M) {
          _(M);
        }
      }
      function E($) {
        try {
          T(C.throw($));
        } catch (M) {
          _(M);
        }
      }
      function T($) {
        $.done ? x($.value) : z($.value).then(w, E);
      }
      T((C = C.apply(v, k || [])).next());
    });
  };
  let { margin: l = !0 } = e;
  const s = Rl([0, 0]);
  Nl(n, s, (v) => t(1, r = v));
  const u = Rl([0, 0]);
  Nl(n, u, (v) => t(2, a = v));
  let h;
  function d() {
    return i(this, void 0, void 0, function* () {
      yield Promise.all([s.set([125, 140]), u.set([-125, -140])]), yield Promise.all([s.set([-125, 140]), u.set([125, -140])]), yield Promise.all([s.set([-125, 0]), u.set([125, -0])]), yield Promise.all([s.set([125, 0]), u.set([-125, 0])]);
    });
  }
  function g() {
    return i(this, void 0, void 0, function* () {
      yield d(), h || g();
    });
  }
  function p() {
    return i(this, void 0, void 0, function* () {
      yield Promise.all([s.set([125, 0]), u.set([-125, 0])]), g();
    });
  }
  return Zd(() => (p(), () => h = !0)), n.$$set = (v) => {
    "margin" in v && t(0, l = v.margin);
  }, [l, r, a, s, u];
}
class Jd extends Gd {
  constructor(e) {
    super(), jd(this, e, Qd, Kd, Xd, { margin: 0 });
  }
}
const {
  SvelteComponent: em,
  append_hydration: b0,
  attr: Dt,
  binding_callbacks: Il,
  check_outros: va,
  children: qt,
  claim_component: tu,
  claim_element: Lt,
  claim_space: lt,
  claim_text: De,
  create_component: ru,
  create_slot: nu,
  destroy_component: au,
  destroy_each: iu,
  detach: X,
  element: It,
  empty: ht,
  ensure_array_like: an,
  get_all_dirty_from_scope: lu,
  get_slot_changes: su,
  group_outros: _a,
  init: tm,
  insert_hydration: te,
  mount_component: ou,
  noop: ba,
  safe_not_equal: rm,
  set_data: dt,
  set_style: h0,
  space: st,
  text: Se,
  toggle_class: at,
  transition_in: kt,
  transition_out: Ot,
  update_slot_base: uu
} = window.__gradio__svelte__internal, { tick: nm } = window.__gradio__svelte__internal, { onDestroy: am } = window.__gradio__svelte__internal, { createEventDispatcher: im } = window.__gradio__svelte__internal, lm = (n) => ({}), Ol = (n) => ({}), sm = (n) => ({}), Pl = (n) => ({});
function Hl(n, e, t) {
  const r = n.slice();
  return r[40] = e[t], r[42] = t, r;
}
function Ul(n, e, t) {
  const r = n.slice();
  return r[40] = e[t], r;
}
function om(n) {
  let e, t, r, a, i = (
    /*i18n*/
    n[1]("common.error") + ""
  ), l, s, u;
  t = new zd({
    props: {
      Icon: Id,
      label: (
        /*i18n*/
        n[1]("common.clear")
      ),
      disabled: !1
    }
  }), t.$on(
    "click",
    /*click_handler*/
    n[32]
  );
  const h = (
    /*#slots*/
    n[30].error
  ), d = nu(
    h,
    n,
    /*$$scope*/
    n[29],
    Ol
  );
  return {
    c() {
      e = It("div"), ru(t.$$.fragment), r = st(), a = It("span"), l = Se(i), s = st(), d && d.c(), this.h();
    },
    l(g) {
      e = Lt(g, "DIV", { class: !0 });
      var p = qt(e);
      tu(t.$$.fragment, p), p.forEach(X), r = lt(g), a = Lt(g, "SPAN", { class: !0 });
      var v = qt(a);
      l = De(v, i), v.forEach(X), s = lt(g), d && d.l(g), this.h();
    },
    h() {
      Dt(e, "class", "clear-status svelte-17v219f"), Dt(a, "class", "error svelte-17v219f");
    },
    m(g, p) {
      te(g, e, p), ou(t, e, null), te(g, r, p), te(g, a, p), b0(a, l), te(g, s, p), d && d.m(g, p), u = !0;
    },
    p(g, p) {
      const v = {};
      p[0] & /*i18n*/
      2 && (v.label = /*i18n*/
      g[1]("common.clear")), t.$set(v), (!u || p[0] & /*i18n*/
      2) && i !== (i = /*i18n*/
      g[1]("common.error") + "") && dt(l, i), d && d.p && (!u || p[0] & /*$$scope*/
      536870912) && uu(
        d,
        h,
        g,
        /*$$scope*/
        g[29],
        u ? su(
          h,
          /*$$scope*/
          g[29],
          p,
          lm
        ) : lu(
          /*$$scope*/
          g[29]
        ),
        Ol
      );
    },
    i(g) {
      u || (kt(t.$$.fragment, g), kt(d, g), u = !0);
    },
    o(g) {
      Ot(t.$$.fragment, g), Ot(d, g), u = !1;
    },
    d(g) {
      g && (X(e), X(r), X(a), X(s)), au(t), d && d.d(g);
    }
  };
}
function um(n) {
  let e, t, r, a, i, l, s, u, h, d = (
    /*variant*/
    n[8] === "default" && /*show_eta_bar*/
    n[18] && /*show_progress*/
    n[6] === "full" && Gl(n)
  );
  function g(_, w) {
    if (
      /*progress*/
      _[7]
    ) return dm;
    if (
      /*queue_position*/
      _[2] !== null && /*queue_size*/
      _[3] !== void 0 && /*queue_position*/
      _[2] >= 0
    ) return hm;
    if (
      /*queue_position*/
      _[2] === 0
    ) return cm;
  }
  let p = g(n), v = p && p(n), k = (
    /*timer*/
    n[5] && jl(n)
  );
  const A = [gm, pm], C = [];
  function z(_, w) {
    return (
      /*last_progress_level*/
      _[15] != null ? 0 : (
        /*show_progress*/
        _[6] === "full" ? 1 : -1
      )
    );
  }
  ~(i = z(n)) && (l = C[i] = A[i](n));
  let x = !/*timer*/
  n[5] && es(n);
  return {
    c() {
      d && d.c(), e = st(), t = It("div"), v && v.c(), r = st(), k && k.c(), a = st(), l && l.c(), s = st(), x && x.c(), u = ht(), this.h();
    },
    l(_) {
      d && d.l(_), e = lt(_), t = Lt(_, "DIV", { class: !0 });
      var w = qt(t);
      v && v.l(w), r = lt(w), k && k.l(w), w.forEach(X), a = lt(_), l && l.l(_), s = lt(_), x && x.l(_), u = ht(), this.h();
    },
    h() {
      Dt(t, "class", "progress-text svelte-17v219f"), at(
        t,
        "meta-text-center",
        /*variant*/
        n[8] === "center"
      ), at(
        t,
        "meta-text",
        /*variant*/
        n[8] === "default"
      );
    },
    m(_, w) {
      d && d.m(_, w), te(_, e, w), te(_, t, w), v && v.m(t, null), b0(t, r), k && k.m(t, null), te(_, a, w), ~i && C[i].m(_, w), te(_, s, w), x && x.m(_, w), te(_, u, w), h = !0;
    },
    p(_, w) {
      /*variant*/
      _[8] === "default" && /*show_eta_bar*/
      _[18] && /*show_progress*/
      _[6] === "full" ? d ? d.p(_, w) : (d = Gl(_), d.c(), d.m(e.parentNode, e)) : d && (d.d(1), d = null), p === (p = g(_)) && v ? v.p(_, w) : (v && v.d(1), v = p && p(_), v && (v.c(), v.m(t, r))), /*timer*/
      _[5] ? k ? k.p(_, w) : (k = jl(_), k.c(), k.m(t, null)) : k && (k.d(1), k = null), (!h || w[0] & /*variant*/
      256) && at(
        t,
        "meta-text-center",
        /*variant*/
        _[8] === "center"
      ), (!h || w[0] & /*variant*/
      256) && at(
        t,
        "meta-text",
        /*variant*/
        _[8] === "default"
      );
      let E = i;
      i = z(_), i === E ? ~i && C[i].p(_, w) : (l && (_a(), Ot(C[E], 1, 1, () => {
        C[E] = null;
      }), va()), ~i ? (l = C[i], l ? l.p(_, w) : (l = C[i] = A[i](_), l.c()), kt(l, 1), l.m(s.parentNode, s)) : l = null), /*timer*/
      _[5] ? x && (_a(), Ot(x, 1, 1, () => {
        x = null;
      }), va()) : x ? (x.p(_, w), w[0] & /*timer*/
      32 && kt(x, 1)) : (x = es(_), x.c(), kt(x, 1), x.m(u.parentNode, u));
    },
    i(_) {
      h || (kt(l), kt(x), h = !0);
    },
    o(_) {
      Ot(l), Ot(x), h = !1;
    },
    d(_) {
      _ && (X(e), X(t), X(a), X(s), X(u)), d && d.d(_), v && v.d(), k && k.d(), ~i && C[i].d(_), x && x.d(_);
    }
  };
}
function Gl(n) {
  let e, t = `translateX(${/*eta_level*/
  (n[17] || 0) * 100 - 100}%)`;
  return {
    c() {
      e = It("div"), this.h();
    },
    l(r) {
      e = Lt(r, "DIV", { class: !0 }), qt(e).forEach(X), this.h();
    },
    h() {
      Dt(e, "class", "eta-bar svelte-17v219f"), h0(e, "transform", t);
    },
    m(r, a) {
      te(r, e, a);
    },
    p(r, a) {
      a[0] & /*eta_level*/
      131072 && t !== (t = `translateX(${/*eta_level*/
      (r[17] || 0) * 100 - 100}%)`) && h0(e, "transform", t);
    },
    d(r) {
      r && X(e);
    }
  };
}
function cm(n) {
  let e;
  return {
    c() {
      e = Se("processing |");
    },
    l(t) {
      e = De(t, "processing |");
    },
    m(t, r) {
      te(t, e, r);
    },
    p: ba,
    d(t) {
      t && X(e);
    }
  };
}
function hm(n) {
  let e, t = (
    /*queue_position*/
    n[2] + 1 + ""
  ), r, a, i, l;
  return {
    c() {
      e = Se("queue: "), r = Se(t), a = Se("/"), i = Se(
        /*queue_size*/
        n[3]
      ), l = Se(" |");
    },
    l(s) {
      e = De(s, "queue: "), r = De(s, t), a = De(s, "/"), i = De(
        s,
        /*queue_size*/
        n[3]
      ), l = De(s, " |");
    },
    m(s, u) {
      te(s, e, u), te(s, r, u), te(s, a, u), te(s, i, u), te(s, l, u);
    },
    p(s, u) {
      u[0] & /*queue_position*/
      4 && t !== (t = /*queue_position*/
      s[2] + 1 + "") && dt(r, t), u[0] & /*queue_size*/
      8 && dt(
        i,
        /*queue_size*/
        s[3]
      );
    },
    d(s) {
      s && (X(e), X(r), X(a), X(i), X(l));
    }
  };
}
function dm(n) {
  let e, t = an(
    /*progress*/
    n[7]
  ), r = [];
  for (let a = 0; a < t.length; a += 1)
    r[a] = Wl(Ul(n, t, a));
  return {
    c() {
      for (let a = 0; a < r.length; a += 1)
        r[a].c();
      e = ht();
    },
    l(a) {
      for (let i = 0; i < r.length; i += 1)
        r[i].l(a);
      e = ht();
    },
    m(a, i) {
      for (let l = 0; l < r.length; l += 1)
        r[l] && r[l].m(a, i);
      te(a, e, i);
    },
    p(a, i) {
      if (i[0] & /*progress*/
      128) {
        t = an(
          /*progress*/
          a[7]
        );
        let l;
        for (l = 0; l < t.length; l += 1) {
          const s = Ul(a, t, l);
          r[l] ? r[l].p(s, i) : (r[l] = Wl(s), r[l].c(), r[l].m(e.parentNode, e));
        }
        for (; l < r.length; l += 1)
          r[l].d(1);
        r.length = t.length;
      }
    },
    d(a) {
      a && X(e), iu(r, a);
    }
  };
}
function Vl(n) {
  let e, t = (
    /*p*/
    n[40].unit + ""
  ), r, a, i = " ", l;
  function s(d, g) {
    return (
      /*p*/
      d[40].length != null ? fm : mm
    );
  }
  let u = s(n), h = u(n);
  return {
    c() {
      h.c(), e = st(), r = Se(t), a = Se(" | "), l = Se(i);
    },
    l(d) {
      h.l(d), e = lt(d), r = De(d, t), a = De(d, " | "), l = De(d, i);
    },
    m(d, g) {
      h.m(d, g), te(d, e, g), te(d, r, g), te(d, a, g), te(d, l, g);
    },
    p(d, g) {
      u === (u = s(d)) && h ? h.p(d, g) : (h.d(1), h = u(d), h && (h.c(), h.m(e.parentNode, e))), g[0] & /*progress*/
      128 && t !== (t = /*p*/
      d[40].unit + "") && dt(r, t);
    },
    d(d) {
      d && (X(e), X(r), X(a), X(l)), h.d(d);
    }
  };
}
function mm(n) {
  let e = B0(
    /*p*/
    n[40].index || 0
  ) + "", t;
  return {
    c() {
      t = Se(e);
    },
    l(r) {
      t = De(r, e);
    },
    m(r, a) {
      te(r, t, a);
    },
    p(r, a) {
      a[0] & /*progress*/
      128 && e !== (e = B0(
        /*p*/
        r[40].index || 0
      ) + "") && dt(t, e);
    },
    d(r) {
      r && X(t);
    }
  };
}
function fm(n) {
  let e = B0(
    /*p*/
    n[40].index || 0
  ) + "", t, r, a = B0(
    /*p*/
    n[40].length
  ) + "", i;
  return {
    c() {
      t = Se(e), r = Se("/"), i = Se(a);
    },
    l(l) {
      t = De(l, e), r = De(l, "/"), i = De(l, a);
    },
    m(l, s) {
      te(l, t, s), te(l, r, s), te(l, i, s);
    },
    p(l, s) {
      s[0] & /*progress*/
      128 && e !== (e = B0(
        /*p*/
        l[40].index || 0
      ) + "") && dt(t, e), s[0] & /*progress*/
      128 && a !== (a = B0(
        /*p*/
        l[40].length
      ) + "") && dt(i, a);
    },
    d(l) {
      l && (X(t), X(r), X(i));
    }
  };
}
function Wl(n) {
  let e, t = (
    /*p*/
    n[40].index != null && Vl(n)
  );
  return {
    c() {
      t && t.c(), e = ht();
    },
    l(r) {
      t && t.l(r), e = ht();
    },
    m(r, a) {
      t && t.m(r, a), te(r, e, a);
    },
    p(r, a) {
      /*p*/
      r[40].index != null ? t ? t.p(r, a) : (t = Vl(r), t.c(), t.m(e.parentNode, e)) : t && (t.d(1), t = null);
    },
    d(r) {
      r && X(e), t && t.d(r);
    }
  };
}
function jl(n) {
  let e, t = (
    /*eta*/
    n[0] ? `/${/*formatted_eta*/
    n[19]}` : ""
  ), r, a;
  return {
    c() {
      e = Se(
        /*formatted_timer*/
        n[20]
      ), r = Se(t), a = Se("s");
    },
    l(i) {
      e = De(
        i,
        /*formatted_timer*/
        n[20]
      ), r = De(i, t), a = De(i, "s");
    },
    m(i, l) {
      te(i, e, l), te(i, r, l), te(i, a, l);
    },
    p(i, l) {
      l[0] & /*formatted_timer*/
      1048576 && dt(
        e,
        /*formatted_timer*/
        i[20]
      ), l[0] & /*eta, formatted_eta*/
      524289 && t !== (t = /*eta*/
      i[0] ? `/${/*formatted_eta*/
      i[19]}` : "") && dt(r, t);
    },
    d(i) {
      i && (X(e), X(r), X(a));
    }
  };
}
function pm(n) {
  let e, t;
  return e = new Jd({
    props: { margin: (
      /*variant*/
      n[8] === "default"
    ) }
  }), {
    c() {
      ru(e.$$.fragment);
    },
    l(r) {
      tu(e.$$.fragment, r);
    },
    m(r, a) {
      ou(e, r, a), t = !0;
    },
    p(r, a) {
      const i = {};
      a[0] & /*variant*/
      256 && (i.margin = /*variant*/
      r[8] === "default"), e.$set(i);
    },
    i(r) {
      t || (kt(e.$$.fragment, r), t = !0);
    },
    o(r) {
      Ot(e.$$.fragment, r), t = !1;
    },
    d(r) {
      au(e, r);
    }
  };
}
function gm(n) {
  let e, t, r, a, i, l = `${/*last_progress_level*/
  n[15] * 100}%`, s = (
    /*progress*/
    n[7] != null && Yl(n)
  );
  return {
    c() {
      e = It("div"), t = It("div"), s && s.c(), r = st(), a = It("div"), i = It("div"), this.h();
    },
    l(u) {
      e = Lt(u, "DIV", { class: !0 });
      var h = qt(e);
      t = Lt(h, "DIV", { class: !0 });
      var d = qt(t);
      s && s.l(d), d.forEach(X), r = lt(h), a = Lt(h, "DIV", { class: !0 });
      var g = qt(a);
      i = Lt(g, "DIV", { class: !0 }), qt(i).forEach(X), g.forEach(X), h.forEach(X), this.h();
    },
    h() {
      Dt(t, "class", "progress-level-inner svelte-17v219f"), Dt(i, "class", "progress-bar svelte-17v219f"), h0(i, "width", l), Dt(a, "class", "progress-bar-wrap svelte-17v219f"), Dt(e, "class", "progress-level svelte-17v219f");
    },
    m(u, h) {
      te(u, e, h), b0(e, t), s && s.m(t, null), b0(e, r), b0(e, a), b0(a, i), n[31](i);
    },
    p(u, h) {
      /*progress*/
      u[7] != null ? s ? s.p(u, h) : (s = Yl(u), s.c(), s.m(t, null)) : s && (s.d(1), s = null), h[0] & /*last_progress_level*/
      32768 && l !== (l = `${/*last_progress_level*/
      u[15] * 100}%`) && h0(i, "width", l);
    },
    i: ba,
    o: ba,
    d(u) {
      u && X(e), s && s.d(), n[31](null);
    }
  };
}
function Yl(n) {
  let e, t = an(
    /*progress*/
    n[7]
  ), r = [];
  for (let a = 0; a < t.length; a += 1)
    r[a] = Jl(Hl(n, t, a));
  return {
    c() {
      for (let a = 0; a < r.length; a += 1)
        r[a].c();
      e = ht();
    },
    l(a) {
      for (let i = 0; i < r.length; i += 1)
        r[i].l(a);
      e = ht();
    },
    m(a, i) {
      for (let l = 0; l < r.length; l += 1)
        r[l] && r[l].m(a, i);
      te(a, e, i);
    },
    p(a, i) {
      if (i[0] & /*progress_level, progress*/
      16512) {
        t = an(
          /*progress*/
          a[7]
        );
        let l;
        for (l = 0; l < t.length; l += 1) {
          const s = Hl(a, t, l);
          r[l] ? r[l].p(s, i) : (r[l] = Jl(s), r[l].c(), r[l].m(e.parentNode, e));
        }
        for (; l < r.length; l += 1)
          r[l].d(1);
        r.length = t.length;
      }
    },
    d(a) {
      a && X(e), iu(r, a);
    }
  };
}
function Xl(n) {
  let e, t, r, a, i = (
    /*i*/
    n[42] !== 0 && vm()
  ), l = (
    /*p*/
    n[40].desc != null && Zl(n)
  ), s = (
    /*p*/
    n[40].desc != null && /*progress_level*/
    n[14] && /*progress_level*/
    n[14][
      /*i*/
      n[42]
    ] != null && Kl()
  ), u = (
    /*progress_level*/
    n[14] != null && Ql(n)
  );
  return {
    c() {
      i && i.c(), e = st(), l && l.c(), t = st(), s && s.c(), r = st(), u && u.c(), a = ht();
    },
    l(h) {
      i && i.l(h), e = lt(h), l && l.l(h), t = lt(h), s && s.l(h), r = lt(h), u && u.l(h), a = ht();
    },
    m(h, d) {
      i && i.m(h, d), te(h, e, d), l && l.m(h, d), te(h, t, d), s && s.m(h, d), te(h, r, d), u && u.m(h, d), te(h, a, d);
    },
    p(h, d) {
      /*p*/
      h[40].desc != null ? l ? l.p(h, d) : (l = Zl(h), l.c(), l.m(t.parentNode, t)) : l && (l.d(1), l = null), /*p*/
      h[40].desc != null && /*progress_level*/
      h[14] && /*progress_level*/
      h[14][
        /*i*/
        h[42]
      ] != null ? s || (s = Kl(), s.c(), s.m(r.parentNode, r)) : s && (s.d(1), s = null), /*progress_level*/
      h[14] != null ? u ? u.p(h, d) : (u = Ql(h), u.c(), u.m(a.parentNode, a)) : u && (u.d(1), u = null);
    },
    d(h) {
      h && (X(e), X(t), X(r), X(a)), i && i.d(h), l && l.d(h), s && s.d(h), u && u.d(h);
    }
  };
}
function vm(n) {
  let e;
  return {
    c() {
      e = Se(" /");
    },
    l(t) {
      e = De(t, " /");
    },
    m(t, r) {
      te(t, e, r);
    },
    d(t) {
      t && X(e);
    }
  };
}
function Zl(n) {
  let e = (
    /*p*/
    n[40].desc + ""
  ), t;
  return {
    c() {
      t = Se(e);
    },
    l(r) {
      t = De(r, e);
    },
    m(r, a) {
      te(r, t, a);
    },
    p(r, a) {
      a[0] & /*progress*/
      128 && e !== (e = /*p*/
      r[40].desc + "") && dt(t, e);
    },
    d(r) {
      r && X(t);
    }
  };
}
function Kl(n) {
  let e;
  return {
    c() {
      e = Se("-");
    },
    l(t) {
      e = De(t, "-");
    },
    m(t, r) {
      te(t, e, r);
    },
    d(t) {
      t && X(e);
    }
  };
}
function Ql(n) {
  let e = (100 * /*progress_level*/
  (n[14][
    /*i*/
    n[42]
  ] || 0)).toFixed(1) + "", t, r;
  return {
    c() {
      t = Se(e), r = Se("%");
    },
    l(a) {
      t = De(a, e), r = De(a, "%");
    },
    m(a, i) {
      te(a, t, i), te(a, r, i);
    },
    p(a, i) {
      i[0] & /*progress_level*/
      16384 && e !== (e = (100 * /*progress_level*/
      (a[14][
        /*i*/
        a[42]
      ] || 0)).toFixed(1) + "") && dt(t, e);
    },
    d(a) {
      a && (X(t), X(r));
    }
  };
}
function Jl(n) {
  let e, t = (
    /*p*/
    (n[40].desc != null || /*progress_level*/
    n[14] && /*progress_level*/
    n[14][
      /*i*/
      n[42]
    ] != null) && Xl(n)
  );
  return {
    c() {
      t && t.c(), e = ht();
    },
    l(r) {
      t && t.l(r), e = ht();
    },
    m(r, a) {
      t && t.m(r, a), te(r, e, a);
    },
    p(r, a) {
      /*p*/
      r[40].desc != null || /*progress_level*/
      r[14] && /*progress_level*/
      r[14][
        /*i*/
        r[42]
      ] != null ? t ? t.p(r, a) : (t = Xl(r), t.c(), t.m(e.parentNode, e)) : t && (t.d(1), t = null);
    },
    d(r) {
      r && X(e), t && t.d(r);
    }
  };
}
function es(n) {
  let e, t, r, a;
  const i = (
    /*#slots*/
    n[30]["additional-loading-text"]
  ), l = nu(
    i,
    n,
    /*$$scope*/
    n[29],
    Pl
  );
  return {
    c() {
      e = It("p"), t = Se(
        /*loading_text*/
        n[9]
      ), r = st(), l && l.c(), this.h();
    },
    l(s) {
      e = Lt(s, "P", { class: !0 });
      var u = qt(e);
      t = De(
        u,
        /*loading_text*/
        n[9]
      ), u.forEach(X), r = lt(s), l && l.l(s), this.h();
    },
    h() {
      Dt(e, "class", "loading svelte-17v219f");
    },
    m(s, u) {
      te(s, e, u), b0(e, t), te(s, r, u), l && l.m(s, u), a = !0;
    },
    p(s, u) {
      (!a || u[0] & /*loading_text*/
      512) && dt(
        t,
        /*loading_text*/
        s[9]
      ), l && l.p && (!a || u[0] & /*$$scope*/
      536870912) && uu(
        l,
        i,
        s,
        /*$$scope*/
        s[29],
        a ? su(
          i,
          /*$$scope*/
          s[29],
          u,
          sm
        ) : lu(
          /*$$scope*/
          s[29]
        ),
        Pl
      );
    },
    i(s) {
      a || (kt(l, s), a = !0);
    },
    o(s) {
      Ot(l, s), a = !1;
    },
    d(s) {
      s && (X(e), X(r)), l && l.d(s);
    }
  };
}
function _m(n) {
  let e, t, r, a, i;
  const l = [um, om], s = [];
  function u(h, d) {
    return (
      /*status*/
      h[4] === "pending" ? 0 : (
        /*status*/
        h[4] === "error" ? 1 : -1
      )
    );
  }
  return ~(t = u(n)) && (r = s[t] = l[t](n)), {
    c() {
      e = It("div"), r && r.c(), this.h();
    },
    l(h) {
      e = Lt(h, "DIV", { class: !0 });
      var d = qt(e);
      r && r.l(d), d.forEach(X), this.h();
    },
    h() {
      Dt(e, "class", a = "wrap " + /*variant*/
      n[8] + " " + /*show_progress*/
      n[6] + " svelte-17v219f"), at(e, "hide", !/*status*/
      n[4] || /*status*/
      n[4] === "complete" || /*show_progress*/
      n[6] === "hidden" || /*status*/
      n[4] == "streaming"), at(
        e,
        "translucent",
        /*variant*/
        n[8] === "center" && /*status*/
        (n[4] === "pending" || /*status*/
        n[4] === "error") || /*translucent*/
        n[11] || /*show_progress*/
        n[6] === "minimal"
      ), at(
        e,
        "generating",
        /*status*/
        n[4] === "generating" && /*show_progress*/
        n[6] === "full"
      ), at(
        e,
        "border",
        /*border*/
        n[12]
      ), h0(
        e,
        "position",
        /*absolute*/
        n[10] ? "absolute" : "static"
      ), h0(
        e,
        "padding",
        /*absolute*/
        n[10] ? "0" : "var(--size-8) 0"
      );
    },
    m(h, d) {
      te(h, e, d), ~t && s[t].m(e, null), n[33](e), i = !0;
    },
    p(h, d) {
      let g = t;
      t = u(h), t === g ? ~t && s[t].p(h, d) : (r && (_a(), Ot(s[g], 1, 1, () => {
        s[g] = null;
      }), va()), ~t ? (r = s[t], r ? r.p(h, d) : (r = s[t] = l[t](h), r.c()), kt(r, 1), r.m(e, null)) : r = null), (!i || d[0] & /*variant, show_progress*/
      320 && a !== (a = "wrap " + /*variant*/
      h[8] + " " + /*show_progress*/
      h[6] + " svelte-17v219f")) && Dt(e, "class", a), (!i || d[0] & /*variant, show_progress, status, show_progress*/
      336) && at(e, "hide", !/*status*/
      h[4] || /*status*/
      h[4] === "complete" || /*show_progress*/
      h[6] === "hidden" || /*status*/
      h[4] == "streaming"), (!i || d[0] & /*variant, show_progress, variant, status, translucent, show_progress*/
      2384) && at(
        e,
        "translucent",
        /*variant*/
        h[8] === "center" && /*status*/
        (h[4] === "pending" || /*status*/
        h[4] === "error") || /*translucent*/
        h[11] || /*show_progress*/
        h[6] === "minimal"
      ), (!i || d[0] & /*variant, show_progress, status, show_progress*/
      336) && at(
        e,
        "generating",
        /*status*/
        h[4] === "generating" && /*show_progress*/
        h[6] === "full"
      ), (!i || d[0] & /*variant, show_progress, border*/
      4416) && at(
        e,
        "border",
        /*border*/
        h[12]
      ), d[0] & /*absolute*/
      1024 && h0(
        e,
        "position",
        /*absolute*/
        h[10] ? "absolute" : "static"
      ), d[0] & /*absolute*/
      1024 && h0(
        e,
        "padding",
        /*absolute*/
        h[10] ? "0" : "var(--size-8) 0"
      );
    },
    i(h) {
      i || (kt(r), i = !0);
    },
    o(h) {
      Ot(r), i = !1;
    },
    d(h) {
      h && X(e), ~t && s[t].d(), n[33](null);
    }
  };
}
var bm = function(n, e, t, r) {
  function a(i) {
    return i instanceof t ? i : new t(function(l) {
      l(i);
    });
  }
  return new (t || (t = Promise))(function(i, l) {
    function s(d) {
      try {
        h(r.next(d));
      } catch (g) {
        l(g);
      }
    }
    function u(d) {
      try {
        h(r.throw(d));
      } catch (g) {
        l(g);
      }
    }
    function h(d) {
      d.done ? i(d.value) : a(d.value).then(s, u);
    }
    h((r = r.apply(n, e || [])).next());
  });
};
let Ir = [], jn = !1;
const ym = typeof window < "u", cu = ym ? window.requestAnimationFrame : (n) => {
};
function wm(n) {
  return bm(this, arguments, void 0, function* (e, t = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && t !== !0)) {
      if (Ir.push(e), !jn) jn = !0;
      else return;
      yield nm(), cu(() => {
        let r = [0, 0];
        for (let a = 0; a < Ir.length; a++) {
          const l = Ir[a].getBoundingClientRect();
          (a === 0 || l.top + window.scrollY <= r[0]) && (r[0] = l.top + window.scrollY, r[1] = a);
        }
        window.scrollTo({ top: r[0] - 20, behavior: "smooth" }), jn = !1, Ir = [];
      });
    }
  });
}
function xm(n, e, t) {
  let r, { $$slots: a = {}, $$scope: i } = e;
  const l = im();
  let { i18n: s } = e, { eta: u = null } = e, { queue_position: h } = e, { queue_size: d } = e, { status: g } = e, { scroll_to_output: p = !1 } = e, { timer: v = !0 } = e, { show_progress: k = "full" } = e, { message: A = null } = e, { progress: C = null } = e, { variant: z = "default" } = e, { loading_text: x = "Loading..." } = e, { absolute: _ = !0 } = e, { translucent: w = !1 } = e, { border: E = !1 } = e, { autoscroll: T } = e, $, M = !1, B = 0, G = 0, U = null, j = null, oe = 0, ee = null, ue, fe = null, Ee = !0;
  const ne = () => {
    t(0, u = t(27, U = t(19, N = null))), t(25, B = performance.now()), t(26, G = 0), M = !0, ve();
  };
  function ve() {
    cu(() => {
      t(26, G = (performance.now() - B) / 1e3), M && ve();
    });
  }
  function we() {
    t(26, G = 0), t(0, u = t(27, U = t(19, N = null))), M && (M = !1);
  }
  am(() => {
    M && we();
  });
  let N = null;
  function se(O) {
    Il[O ? "unshift" : "push"](() => {
      fe = O, t(16, fe), t(7, C), t(14, ee), t(15, ue);
    });
  }
  const ce = () => {
    l("clear_status");
  };
  function Ce(O) {
    Il[O ? "unshift" : "push"](() => {
      $ = O, t(13, $);
    });
  }
  return n.$$set = (O) => {
    "i18n" in O && t(1, s = O.i18n), "eta" in O && t(0, u = O.eta), "queue_position" in O && t(2, h = O.queue_position), "queue_size" in O && t(3, d = O.queue_size), "status" in O && t(4, g = O.status), "scroll_to_output" in O && t(22, p = O.scroll_to_output), "timer" in O && t(5, v = O.timer), "show_progress" in O && t(6, k = O.show_progress), "message" in O && t(23, A = O.message), "progress" in O && t(7, C = O.progress), "variant" in O && t(8, z = O.variant), "loading_text" in O && t(9, x = O.loading_text), "absolute" in O && t(10, _ = O.absolute), "translucent" in O && t(11, w = O.translucent), "border" in O && t(12, E = O.border), "autoscroll" in O && t(24, T = O.autoscroll), "$$scope" in O && t(29, i = O.$$scope);
  }, n.$$.update = () => {
    n.$$.dirty[0] & /*eta, old_eta, timer_start, eta_from_start*/
    436207617 && (u === null && t(0, u = U), u != null && U !== u && (t(28, j = (performance.now() - B) / 1e3 + u), t(19, N = j.toFixed(1)), t(27, U = u))), n.$$.dirty[0] & /*eta_from_start, timer_diff*/
    335544320 && t(17, oe = j === null || j <= 0 || !G ? null : Math.min(G / j, 1)), n.$$.dirty[0] & /*progress*/
    128 && C != null && t(18, Ee = !1), n.$$.dirty[0] & /*progress, progress_level, progress_bar, last_progress_level*/
    114816 && (C != null ? t(14, ee = C.map((O) => {
      if (O.index != null && O.length != null)
        return O.index / O.length;
      if (O.progress != null)
        return O.progress;
    })) : t(14, ee = null), ee ? (t(15, ue = ee[ee.length - 1]), fe && (ue === 0 ? t(16, fe.style.transition = "0", fe) : t(16, fe.style.transition = "150ms", fe))) : t(15, ue = void 0)), n.$$.dirty[0] & /*status*/
    16 && (g === "pending" ? ne() : we()), n.$$.dirty[0] & /*el, scroll_to_output, status, autoscroll*/
    20979728 && $ && p && (g === "pending" || g === "complete") && wm($, T), n.$$.dirty[0] & /*status, message*/
    8388624, n.$$.dirty[0] & /*timer_diff*/
    67108864 && t(20, r = G.toFixed(1));
  }, [
    u,
    s,
    h,
    d,
    g,
    v,
    k,
    C,
    z,
    x,
    _,
    w,
    E,
    $,
    ee,
    ue,
    fe,
    oe,
    Ee,
    N,
    r,
    l,
    p,
    A,
    T,
    B,
    G,
    U,
    j,
    i,
    a,
    se,
    ce,
    Ce
  ];
}
class km extends em {
  constructor(e) {
    super(), tm(
      this,
      e,
      xm,
      _m,
      rm,
      {
        i18n: 1,
        eta: 0,
        queue_position: 2,
        queue_size: 3,
        status: 4,
        scroll_to_output: 22,
        timer: 5,
        show_progress: 6,
        message: 23,
        progress: 7,
        variant: 8,
        loading_text: 9,
        absolute: 10,
        translucent: 11,
        border: 12,
        autoscroll: 24
      },
      null,
      [-1, -1]
    );
  }
}
/*! @license DOMPurify 3.2.6 | (c) Cure53 and other contributors | Released under the Apache license 2.0 and Mozilla Public License 2.0 | github.com/cure53/DOMPurify/blob/3.2.6/LICENSE */
const {
  entries: hu,
  setPrototypeOf: ts,
  isFrozen: Dm,
  getPrototypeOf: Sm,
  getOwnPropertyDescriptor: Am
} = Object;
let {
  freeze: Ge,
  seal: mt,
  create: du
} = Object, {
  apply: ya,
  construct: wa
} = typeof Reflect < "u" && Reflect;
Ge || (Ge = function(e) {
  return e;
});
mt || (mt = function(e) {
  return e;
});
ya || (ya = function(e, t, r) {
  return e.apply(t, r);
});
wa || (wa = function(e, t) {
  return new e(...t);
});
const Or = Ve(Array.prototype.forEach), Em = Ve(Array.prototype.lastIndexOf), rs = Ve(Array.prototype.pop), W0 = Ve(Array.prototype.push), Fm = Ve(Array.prototype.splice), Yr = Ve(String.prototype.toLowerCase), Yn = Ve(String.prototype.toString), ns = Ve(String.prototype.match), j0 = Ve(String.prototype.replace), Cm = Ve(String.prototype.indexOf), Tm = Ve(String.prototype.trim), wt = Ve(Object.prototype.hasOwnProperty), He = Ve(RegExp.prototype.test), Y0 = $m(TypeError);
function Ve(n) {
  return function(e) {
    e instanceof RegExp && (e.lastIndex = 0);
    for (var t = arguments.length, r = new Array(t > 1 ? t - 1 : 0), a = 1; a < t; a++)
      r[a - 1] = arguments[a];
    return ya(n, e, r);
  };
}
function $m(n) {
  return function() {
    for (var e = arguments.length, t = new Array(e), r = 0; r < e; r++)
      t[r] = arguments[r];
    return wa(n, t);
  };
}
function ae(n, e) {
  let t = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : Yr;
  ts && ts(n, null);
  let r = e.length;
  for (; r--; ) {
    let a = e[r];
    if (typeof a == "string") {
      const i = t(a);
      i !== a && (Dm(e) || (e[r] = i), a = i);
    }
    n[a] = !0;
  }
  return n;
}
function Mm(n) {
  for (let e = 0; e < n.length; e++)
    wt(n, e) || (n[e] = null);
  return n;
}
function Qt(n) {
  const e = du(null);
  for (const [t, r] of hu(n))
    wt(n, t) && (Array.isArray(r) ? e[t] = Mm(r) : r && typeof r == "object" && r.constructor === Object ? e[t] = Qt(r) : e[t] = r);
  return e;
}
function X0(n, e) {
  for (; n !== null; ) {
    const r = Am(n, e);
    if (r) {
      if (r.get)
        return Ve(r.get);
      if (typeof r.value == "function")
        return Ve(r.value);
    }
    n = Sm(n);
  }
  function t() {
    return null;
  }
  return t;
}
const as = Ge(["a", "abbr", "acronym", "address", "area", "article", "aside", "audio", "b", "bdi", "bdo", "big", "blink", "blockquote", "body", "br", "button", "canvas", "caption", "center", "cite", "code", "col", "colgroup", "content", "data", "datalist", "dd", "decorator", "del", "details", "dfn", "dialog", "dir", "div", "dl", "dt", "element", "em", "fieldset", "figcaption", "figure", "font", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "i", "img", "input", "ins", "kbd", "label", "legend", "li", "main", "map", "mark", "marquee", "menu", "menuitem", "meter", "nav", "nobr", "ol", "optgroup", "option", "output", "p", "picture", "pre", "progress", "q", "rp", "rt", "ruby", "s", "samp", "section", "select", "shadow", "small", "source", "spacer", "span", "strike", "strong", "style", "sub", "summary", "sup", "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "time", "tr", "track", "tt", "u", "ul", "var", "video", "wbr"]), Xn = Ge(["svg", "a", "altglyph", "altglyphdef", "altglyphitem", "animatecolor", "animatemotion", "animatetransform", "circle", "clippath", "defs", "desc", "ellipse", "filter", "font", "g", "glyph", "glyphref", "hkern", "image", "line", "lineargradient", "marker", "mask", "metadata", "mpath", "path", "pattern", "polygon", "polyline", "radialgradient", "rect", "stop", "style", "switch", "symbol", "text", "textpath", "title", "tref", "tspan", "view", "vkern"]), Zn = Ge(["feBlend", "feColorMatrix", "feComponentTransfer", "feComposite", "feConvolveMatrix", "feDiffuseLighting", "feDisplacementMap", "feDistantLight", "feDropShadow", "feFlood", "feFuncA", "feFuncB", "feFuncG", "feFuncR", "feGaussianBlur", "feImage", "feMerge", "feMergeNode", "feMorphology", "feOffset", "fePointLight", "feSpecularLighting", "feSpotLight", "feTile", "feTurbulence"]), zm = Ge(["animate", "color-profile", "cursor", "discard", "font-face", "font-face-format", "font-face-name", "font-face-src", "font-face-uri", "foreignobject", "hatch", "hatchpath", "mesh", "meshgradient", "meshpatch", "meshrow", "missing-glyph", "script", "set", "solidcolor", "unknown", "use"]), Kn = Ge(["math", "menclose", "merror", "mfenced", "mfrac", "mglyph", "mi", "mlabeledtr", "mmultiscripts", "mn", "mo", "mover", "mpadded", "mphantom", "mroot", "mrow", "ms", "mspace", "msqrt", "mstyle", "msub", "msup", "msubsup", "mtable", "mtd", "mtext", "mtr", "munder", "munderover", "mprescripts"]), Bm = Ge(["maction", "maligngroup", "malignmark", "mlongdiv", "mscarries", "mscarry", "msgroup", "mstack", "msline", "msrow", "semantics", "annotation", "annotation-xml", "mprescripts", "none"]), is = Ge(["#text"]), ls = Ge(["accept", "action", "align", "alt", "autocapitalize", "autocomplete", "autopictureinpicture", "autoplay", "background", "bgcolor", "border", "capture", "cellpadding", "cellspacing", "checked", "cite", "class", "clear", "color", "cols", "colspan", "controls", "controlslist", "coords", "crossorigin", "datetime", "decoding", "default", "dir", "disabled", "disablepictureinpicture", "disableremoteplayback", "download", "draggable", "enctype", "enterkeyhint", "face", "for", "headers", "height", "hidden", "high", "href", "hreflang", "id", "inputmode", "integrity", "ismap", "kind", "label", "lang", "list", "loading", "loop", "low", "max", "maxlength", "media", "method", "min", "minlength", "multiple", "muted", "name", "nonce", "noshade", "novalidate", "nowrap", "open", "optimum", "pattern", "placeholder", "playsinline", "popover", "popovertarget", "popovertargetaction", "poster", "preload", "pubdate", "radiogroup", "readonly", "rel", "required", "rev", "reversed", "role", "rows", "rowspan", "spellcheck", "scope", "selected", "shape", "size", "sizes", "span", "srclang", "start", "src", "srcset", "step", "style", "summary", "tabindex", "title", "translate", "type", "usemap", "valign", "value", "width", "wrap", "xmlns", "slot"]), Qn = Ge(["accent-height", "accumulate", "additive", "alignment-baseline", "amplitude", "ascent", "attributename", "attributetype", "azimuth", "basefrequency", "baseline-shift", "begin", "bias", "by", "class", "clip", "clippathunits", "clip-path", "clip-rule", "color", "color-interpolation", "color-interpolation-filters", "color-profile", "color-rendering", "cx", "cy", "d", "dx", "dy", "diffuseconstant", "direction", "display", "divisor", "dur", "edgemode", "elevation", "end", "exponent", "fill", "fill-opacity", "fill-rule", "filter", "filterunits", "flood-color", "flood-opacity", "font-family", "font-size", "font-size-adjust", "font-stretch", "font-style", "font-variant", "font-weight", "fx", "fy", "g1", "g2", "glyph-name", "glyphref", "gradientunits", "gradienttransform", "height", "href", "id", "image-rendering", "in", "in2", "intercept", "k", "k1", "k2", "k3", "k4", "kerning", "keypoints", "keysplines", "keytimes", "lang", "lengthadjust", "letter-spacing", "kernelmatrix", "kernelunitlength", "lighting-color", "local", "marker-end", "marker-mid", "marker-start", "markerheight", "markerunits", "markerwidth", "maskcontentunits", "maskunits", "max", "mask", "media", "method", "mode", "min", "name", "numoctaves", "offset", "operator", "opacity", "order", "orient", "orientation", "origin", "overflow", "paint-order", "path", "pathlength", "patterncontentunits", "patterntransform", "patternunits", "points", "preservealpha", "preserveaspectratio", "primitiveunits", "r", "rx", "ry", "radius", "refx", "refy", "repeatcount", "repeatdur", "restart", "result", "rotate", "scale", "seed", "shape-rendering", "slope", "specularconstant", "specularexponent", "spreadmethod", "startoffset", "stddeviation", "stitchtiles", "stop-color", "stop-opacity", "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "stroke-opacity", "stroke", "stroke-width", "style", "surfacescale", "systemlanguage", "tabindex", "tablevalues", "targetx", "targety", "transform", "transform-origin", "text-anchor", "text-decoration", "text-rendering", "textlength", "type", "u1", "u2", "unicode", "values", "viewbox", "visibility", "version", "vert-adv-y", "vert-origin-x", "vert-origin-y", "width", "word-spacing", "wrap", "writing-mode", "xchannelselector", "ychannelselector", "x", "x1", "x2", "xmlns", "y", "y1", "y2", "z", "zoomandpan"]), ss = Ge(["accent", "accentunder", "align", "bevelled", "close", "columnsalign", "columnlines", "columnspan", "denomalign", "depth", "dir", "display", "displaystyle", "encoding", "fence", "frame", "height", "href", "id", "largeop", "length", "linethickness", "lspace", "lquote", "mathbackground", "mathcolor", "mathsize", "mathvariant", "maxsize", "minsize", "movablelimits", "notation", "numalign", "open", "rowalign", "rowlines", "rowspacing", "rowspan", "rspace", "rquote", "scriptlevel", "scriptminsize", "scriptsizemultiplier", "selection", "separator", "separators", "stretchy", "subscriptshift", "supscriptshift", "symmetric", "voffset", "width", "xmlns"]), Pr = Ge(["xlink:href", "xml:id", "xlink:title", "xml:space", "xmlns:xlink"]), Rm = mt(/\{\{[\w\W]*|[\w\W]*\}\}/gm), Nm = mt(/<%[\w\W]*|[\w\W]*%>/gm), qm = mt(/\$\{[\w\W]*/gm), Lm = mt(/^data-[\-\w.\u00B7-\uFFFF]+$/), Im = mt(/^aria-[\-\w]+$/), mu = mt(
  /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|sms|cid|xmpp|matrix):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i
  // eslint-disable-line no-useless-escape
), Om = mt(/^(?:\w+script|data):/i), Pm = mt(
  /[\u0000-\u0020\u00A0\u1680\u180E\u2000-\u2029\u205F\u3000]/g
  // eslint-disable-line no-control-regex
), fu = mt(/^html$/i), Hm = mt(/^[a-z][.\w]*(-[.\w]+)+$/i);
var os = /* @__PURE__ */ Object.freeze({
  __proto__: null,
  ARIA_ATTR: Im,
  ATTR_WHITESPACE: Pm,
  CUSTOM_ELEMENT: Hm,
  DATA_ATTR: Lm,
  DOCTYPE_NAME: fu,
  ERB_EXPR: Nm,
  IS_ALLOWED_URI: mu,
  IS_SCRIPT_OR_DATA: Om,
  MUSTACHE_EXPR: Rm,
  TMPLIT_EXPR: qm
});
const Z0 = {
  element: 1,
  text: 3,
  // Deprecated
  progressingInstruction: 7,
  comment: 8,
  document: 9
}, Um = function() {
  return typeof window > "u" ? null : window;
}, Gm = function(e, t) {
  if (typeof e != "object" || typeof e.createPolicy != "function")
    return null;
  let r = null;
  const a = "data-tt-policy-suffix";
  t && t.hasAttribute(a) && (r = t.getAttribute(a));
  const i = "dompurify" + (r ? "#" + r : "");
  try {
    return e.createPolicy(i, {
      createHTML(l) {
        return l;
      },
      createScriptURL(l) {
        return l;
      }
    });
  } catch {
    return console.warn("TrustedTypes policy " + i + " could not be created."), null;
  }
}, us = function() {
  return {
    afterSanitizeAttributes: [],
    afterSanitizeElements: [],
    afterSanitizeShadowDOM: [],
    beforeSanitizeAttributes: [],
    beforeSanitizeElements: [],
    beforeSanitizeShadowDOM: [],
    uponSanitizeAttribute: [],
    uponSanitizeElement: [],
    uponSanitizeShadowNode: []
  };
};
function pu() {
  let n = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : Um();
  const e = (Y) => pu(Y);
  if (e.version = "3.2.6", e.removed = [], !n || !n.document || n.document.nodeType !== Z0.document || !n.Element)
    return e.isSupported = !1, e;
  let {
    document: t
  } = n;
  const r = t, a = r.currentScript, {
    DocumentFragment: i,
    HTMLTemplateElement: l,
    Node: s,
    Element: u,
    NodeFilter: h,
    NamedNodeMap: d = n.NamedNodeMap || n.MozNamedAttrMap,
    HTMLFormElement: g,
    DOMParser: p,
    trustedTypes: v
  } = n, k = u.prototype, A = X0(k, "cloneNode"), C = X0(k, "remove"), z = X0(k, "nextSibling"), x = X0(k, "childNodes"), _ = X0(k, "parentNode");
  if (typeof l == "function") {
    const Y = t.createElement("template");
    Y.content && Y.content.ownerDocument && (t = Y.content.ownerDocument);
  }
  let w, E = "";
  const {
    implementation: T,
    createNodeIterator: $,
    createDocumentFragment: M,
    getElementsByTagName: B
  } = t, {
    importNode: G
  } = r;
  let U = us();
  e.isSupported = typeof hu == "function" && typeof _ == "function" && T && T.createHTMLDocument !== void 0;
  const {
    MUSTACHE_EXPR: j,
    ERB_EXPR: oe,
    TMPLIT_EXPR: ee,
    DATA_ATTR: ue,
    ARIA_ATTR: fe,
    IS_SCRIPT_OR_DATA: Ee,
    ATTR_WHITESPACE: ne,
    CUSTOM_ELEMENT: ve
  } = os;
  let {
    IS_ALLOWED_URI: we
  } = os, N = null;
  const se = ae({}, [...as, ...Xn, ...Zn, ...Kn, ...is]);
  let ce = null;
  const Ce = ae({}, [...ls, ...Qn, ...ss, ...Pr]);
  let O = Object.seal(du(null, {
    tagNameCheck: {
      writable: !0,
      configurable: !1,
      enumerable: !0,
      value: null
    },
    attributeNameCheck: {
      writable: !0,
      configurable: !1,
      enumerable: !0,
      value: null
    },
    allowCustomizedBuiltInElements: {
      writable: !0,
      configurable: !1,
      enumerable: !0,
      value: !1
    }
  })), Ie = null, Oe = null, Ke = !0, ft = !0, pt = !1, Gt = !0, At = !1, Et = !0, gt = !1, D0 = !1, S0 = !1, Ft = !1, Vt = !1, Wt = !1, ti = !0, ri = !1;
  const gu = "user-content-";
  let pn = !0, P0 = !1, A0 = {}, E0 = null;
  const ni = ae({}, ["annotation-xml", "audio", "colgroup", "desc", "foreignobject", "head", "iframe", "math", "mi", "mn", "mo", "ms", "mtext", "noembed", "noframes", "noscript", "plaintext", "script", "style", "svg", "template", "thead", "title", "video", "xmp"]);
  let ai = null;
  const ii = ae({}, ["audio", "video", "img", "source", "image", "track"]);
  let gn = null;
  const li = ae({}, ["alt", "class", "for", "id", "label", "name", "pattern", "placeholder", "role", "summary", "title", "value", "style", "xmlns"]), fr = "http://www.w3.org/1998/Math/MathML", pr = "http://www.w3.org/2000/svg", jt = "http://www.w3.org/1999/xhtml";
  let F0 = jt, vn = !1, _n = null;
  const vu = ae({}, [fr, pr, jt], Yn);
  let gr = ae({}, ["mi", "mo", "mn", "ms", "mtext"]), vr = ae({}, ["annotation-xml"]);
  const _u = ae({}, ["title", "style", "font", "a", "script"]);
  let H0 = null;
  const bu = ["application/xhtml+xml", "text/html"], yu = "text/html";
  let ze = null, C0 = null;
  const wu = t.createElement("form"), si = function(S) {
    return S instanceof RegExp || S instanceof Function;
  }, bn = function() {
    let S = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
    if (!(C0 && C0 === S)) {
      if ((!S || typeof S != "object") && (S = {}), S = Qt(S), H0 = // eslint-disable-next-line unicorn/prefer-includes
      bu.indexOf(S.PARSER_MEDIA_TYPE) === -1 ? yu : S.PARSER_MEDIA_TYPE, ze = H0 === "application/xhtml+xml" ? Yn : Yr, N = wt(S, "ALLOWED_TAGS") ? ae({}, S.ALLOWED_TAGS, ze) : se, ce = wt(S, "ALLOWED_ATTR") ? ae({}, S.ALLOWED_ATTR, ze) : Ce, _n = wt(S, "ALLOWED_NAMESPACES") ? ae({}, S.ALLOWED_NAMESPACES, Yn) : vu, gn = wt(S, "ADD_URI_SAFE_ATTR") ? ae(Qt(li), S.ADD_URI_SAFE_ATTR, ze) : li, ai = wt(S, "ADD_DATA_URI_TAGS") ? ae(Qt(ii), S.ADD_DATA_URI_TAGS, ze) : ii, E0 = wt(S, "FORBID_CONTENTS") ? ae({}, S.FORBID_CONTENTS, ze) : ni, Ie = wt(S, "FORBID_TAGS") ? ae({}, S.FORBID_TAGS, ze) : Qt({}), Oe = wt(S, "FORBID_ATTR") ? ae({}, S.FORBID_ATTR, ze) : Qt({}), A0 = wt(S, "USE_PROFILES") ? S.USE_PROFILES : !1, Ke = S.ALLOW_ARIA_ATTR !== !1, ft = S.ALLOW_DATA_ATTR !== !1, pt = S.ALLOW_UNKNOWN_PROTOCOLS || !1, Gt = S.ALLOW_SELF_CLOSE_IN_ATTR !== !1, At = S.SAFE_FOR_TEMPLATES || !1, Et = S.SAFE_FOR_XML !== !1, gt = S.WHOLE_DOCUMENT || !1, Ft = S.RETURN_DOM || !1, Vt = S.RETURN_DOM_FRAGMENT || !1, Wt = S.RETURN_TRUSTED_TYPE || !1, S0 = S.FORCE_BODY || !1, ti = S.SANITIZE_DOM !== !1, ri = S.SANITIZE_NAMED_PROPS || !1, pn = S.KEEP_CONTENT !== !1, P0 = S.IN_PLACE || !1, we = S.ALLOWED_URI_REGEXP || mu, F0 = S.NAMESPACE || jt, gr = S.MATHML_TEXT_INTEGRATION_POINTS || gr, vr = S.HTML_INTEGRATION_POINTS || vr, O = S.CUSTOM_ELEMENT_HANDLING || {}, S.CUSTOM_ELEMENT_HANDLING && si(S.CUSTOM_ELEMENT_HANDLING.tagNameCheck) && (O.tagNameCheck = S.CUSTOM_ELEMENT_HANDLING.tagNameCheck), S.CUSTOM_ELEMENT_HANDLING && si(S.CUSTOM_ELEMENT_HANDLING.attributeNameCheck) && (O.attributeNameCheck = S.CUSTOM_ELEMENT_HANDLING.attributeNameCheck), S.CUSTOM_ELEMENT_HANDLING && typeof S.CUSTOM_ELEMENT_HANDLING.allowCustomizedBuiltInElements == "boolean" && (O.allowCustomizedBuiltInElements = S.CUSTOM_ELEMENT_HANDLING.allowCustomizedBuiltInElements), At && (ft = !1), Vt && (Ft = !0), A0 && (N = ae({}, is), ce = [], A0.html === !0 && (ae(N, as), ae(ce, ls)), A0.svg === !0 && (ae(N, Xn), ae(ce, Qn), ae(ce, Pr)), A0.svgFilters === !0 && (ae(N, Zn), ae(ce, Qn), ae(ce, Pr)), A0.mathMl === !0 && (ae(N, Kn), ae(ce, ss), ae(ce, Pr))), S.ADD_TAGS && (N === se && (N = Qt(N)), ae(N, S.ADD_TAGS, ze)), S.ADD_ATTR && (ce === Ce && (ce = Qt(ce)), ae(ce, S.ADD_ATTR, ze)), S.ADD_URI_SAFE_ATTR && ae(gn, S.ADD_URI_SAFE_ATTR, ze), S.FORBID_CONTENTS && (E0 === ni && (E0 = Qt(E0)), ae(E0, S.FORBID_CONTENTS, ze)), pn && (N["#text"] = !0), gt && ae(N, ["html", "head", "body"]), N.table && (ae(N, ["tbody"]), delete Ie.tbody), S.TRUSTED_TYPES_POLICY) {
        if (typeof S.TRUSTED_TYPES_POLICY.createHTML != "function")
          throw Y0('TRUSTED_TYPES_POLICY configuration option must provide a "createHTML" hook.');
        if (typeof S.TRUSTED_TYPES_POLICY.createScriptURL != "function")
          throw Y0('TRUSTED_TYPES_POLICY configuration option must provide a "createScriptURL" hook.');
        w = S.TRUSTED_TYPES_POLICY, E = w.createHTML("");
      } else
        w === void 0 && (w = Gm(v, a)), w !== null && typeof E == "string" && (E = w.createHTML(""));
      Ge && Ge(S), C0 = S;
    }
  }, oi = ae({}, [...Xn, ...Zn, ...zm]), ui = ae({}, [...Kn, ...Bm]), xu = function(S) {
    let I = _(S);
    (!I || !I.tagName) && (I = {
      namespaceURI: F0,
      tagName: "template"
    });
    const W = Yr(S.tagName), be = Yr(I.tagName);
    return _n[S.namespaceURI] ? S.namespaceURI === pr ? I.namespaceURI === jt ? W === "svg" : I.namespaceURI === fr ? W === "svg" && (be === "annotation-xml" || gr[be]) : !!oi[W] : S.namespaceURI === fr ? I.namespaceURI === jt ? W === "math" : I.namespaceURI === pr ? W === "math" && vr[be] : !!ui[W] : S.namespaceURI === jt ? I.namespaceURI === pr && !vr[be] || I.namespaceURI === fr && !gr[be] ? !1 : !ui[W] && (_u[W] || !oi[W]) : !!(H0 === "application/xhtml+xml" && _n[S.namespaceURI]) : !1;
  }, Ct = function(S) {
    W0(e.removed, {
      element: S
    });
    try {
      _(S).removeChild(S);
    } catch {
      C(S);
    }
  }, T0 = function(S, I) {
    try {
      W0(e.removed, {
        attribute: I.getAttributeNode(S),
        from: I
      });
    } catch {
      W0(e.removed, {
        attribute: null,
        from: I
      });
    }
    if (I.removeAttribute(S), S === "is")
      if (Ft || Vt)
        try {
          Ct(I);
        } catch {
        }
      else
        try {
          I.setAttribute(S, "");
        } catch {
        }
  }, ci = function(S) {
    let I = null, W = null;
    if (S0)
      S = "<remove></remove>" + S;
    else {
      const $e = ns(S, /^[\r\n\t ]+/);
      W = $e && $e[0];
    }
    H0 === "application/xhtml+xml" && F0 === jt && (S = '<html xmlns="http://www.w3.org/1999/xhtml"><head></head><body>' + S + "</body></html>");
    const be = w ? w.createHTML(S) : S;
    if (F0 === jt)
      try {
        I = new p().parseFromString(be, H0);
      } catch {
      }
    if (!I || !I.documentElement) {
      I = T.createDocument(F0, "template", null);
      try {
        I.documentElement.innerHTML = vn ? E : be;
      } catch {
      }
    }
    const Re = I.body || I.documentElement;
    return S && W && Re.insertBefore(t.createTextNode(W), Re.childNodes[0] || null), F0 === jt ? B.call(I, gt ? "html" : "body")[0] : gt ? I.documentElement : Re;
  }, hi = function(S) {
    return $.call(
      S.ownerDocument || S,
      S,
      // eslint-disable-next-line no-bitwise
      h.SHOW_ELEMENT | h.SHOW_COMMENT | h.SHOW_TEXT | h.SHOW_PROCESSING_INSTRUCTION | h.SHOW_CDATA_SECTION,
      null
    );
  }, yn = function(S) {
    return S instanceof g && (typeof S.nodeName != "string" || typeof S.textContent != "string" || typeof S.removeChild != "function" || !(S.attributes instanceof d) || typeof S.removeAttribute != "function" || typeof S.setAttribute != "function" || typeof S.namespaceURI != "string" || typeof S.insertBefore != "function" || typeof S.hasChildNodes != "function");
  }, di = function(S) {
    return typeof s == "function" && S instanceof s;
  };
  function Yt(Y, S, I) {
    Or(Y, (W) => {
      W.call(e, S, I, C0);
    });
  }
  const mi = function(S) {
    let I = null;
    if (Yt(U.beforeSanitizeElements, S, null), yn(S))
      return Ct(S), !0;
    const W = ze(S.nodeName);
    if (Yt(U.uponSanitizeElement, S, {
      tagName: W,
      allowedTags: N
    }), Et && S.hasChildNodes() && !di(S.firstElementChild) && He(/<[/\w!]/g, S.innerHTML) && He(/<[/\w!]/g, S.textContent) || S.nodeType === Z0.progressingInstruction || Et && S.nodeType === Z0.comment && He(/<[/\w]/g, S.data))
      return Ct(S), !0;
    if (!N[W] || Ie[W]) {
      if (!Ie[W] && pi(W) && (O.tagNameCheck instanceof RegExp && He(O.tagNameCheck, W) || O.tagNameCheck instanceof Function && O.tagNameCheck(W)))
        return !1;
      if (pn && !E0[W]) {
        const be = _(S) || S.parentNode, Re = x(S) || S.childNodes;
        if (Re && be) {
          const $e = Re.length;
          for (let je = $e - 1; je >= 0; --je) {
            const Xt = A(Re[je], !0);
            Xt.__removalCount = (S.__removalCount || 0) + 1, be.insertBefore(Xt, z(S));
          }
        }
      }
      return Ct(S), !0;
    }
    return S instanceof u && !xu(S) || (W === "noscript" || W === "noembed" || W === "noframes") && He(/<\/no(script|embed|frames)/i, S.innerHTML) ? (Ct(S), !0) : (At && S.nodeType === Z0.text && (I = S.textContent, Or([j, oe, ee], (be) => {
      I = j0(I, be, " ");
    }), S.textContent !== I && (W0(e.removed, {
      element: S.cloneNode()
    }), S.textContent = I)), Yt(U.afterSanitizeElements, S, null), !1);
  }, fi = function(S, I, W) {
    if (ti && (I === "id" || I === "name") && (W in t || W in wu))
      return !1;
    if (!(ft && !Oe[I] && He(ue, I))) {
      if (!(Ke && He(fe, I))) {
        if (!ce[I] || Oe[I]) {
          if (
            // First condition does a very basic check if a) it's basically a valid custom element tagname AND
            // b) if the tagName passes whatever the user has configured for CUSTOM_ELEMENT_HANDLING.tagNameCheck
            // and c) if the attribute name passes whatever the user has configured for CUSTOM_ELEMENT_HANDLING.attributeNameCheck
            !(pi(S) && (O.tagNameCheck instanceof RegExp && He(O.tagNameCheck, S) || O.tagNameCheck instanceof Function && O.tagNameCheck(S)) && (O.attributeNameCheck instanceof RegExp && He(O.attributeNameCheck, I) || O.attributeNameCheck instanceof Function && O.attributeNameCheck(I)) || // Alternative, second condition checks if it's an `is`-attribute, AND
            // the value passes whatever the user has configured for CUSTOM_ELEMENT_HANDLING.tagNameCheck
            I === "is" && O.allowCustomizedBuiltInElements && (O.tagNameCheck instanceof RegExp && He(O.tagNameCheck, W) || O.tagNameCheck instanceof Function && O.tagNameCheck(W)))
          ) return !1;
        } else if (!gn[I]) {
          if (!He(we, j0(W, ne, ""))) {
            if (!((I === "src" || I === "xlink:href" || I === "href") && S !== "script" && Cm(W, "data:") === 0 && ai[S])) {
              if (!(pt && !He(Ee, j0(W, ne, "")))) {
                if (W)
                  return !1;
              }
            }
          }
        }
      }
    }
    return !0;
  }, pi = function(S) {
    return S !== "annotation-xml" && ns(S, ve);
  }, gi = function(S) {
    Yt(U.beforeSanitizeAttributes, S, null);
    const {
      attributes: I
    } = S;
    if (!I || yn(S))
      return;
    const W = {
      attrName: "",
      attrValue: "",
      keepAttr: !0,
      allowedAttributes: ce,
      forceKeepAttr: void 0
    };
    let be = I.length;
    for (; be--; ) {
      const Re = I[be], {
        name: $e,
        namespaceURI: je,
        value: Xt
      } = Re, U0 = ze($e), wn = Xt;
      let Ne = $e === "value" ? wn : Tm(wn);
      if (W.attrName = U0, W.attrValue = Ne, W.keepAttr = !0, W.forceKeepAttr = void 0, Yt(U.uponSanitizeAttribute, S, W), Ne = W.attrValue, ri && (U0 === "id" || U0 === "name") && (T0($e, S), Ne = gu + Ne), Et && He(/((--!?|])>)|<\/(style|title)/i, Ne)) {
        T0($e, S);
        continue;
      }
      if (W.forceKeepAttr)
        continue;
      if (!W.keepAttr) {
        T0($e, S);
        continue;
      }
      if (!Gt && He(/\/>/i, Ne)) {
        T0($e, S);
        continue;
      }
      At && Or([j, oe, ee], (_i) => {
        Ne = j0(Ne, _i, " ");
      });
      const vi = ze(S.nodeName);
      if (!fi(vi, U0, Ne)) {
        T0($e, S);
        continue;
      }
      if (w && typeof v == "object" && typeof v.getAttributeType == "function" && !je)
        switch (v.getAttributeType(vi, U0)) {
          case "TrustedHTML": {
            Ne = w.createHTML(Ne);
            break;
          }
          case "TrustedScriptURL": {
            Ne = w.createScriptURL(Ne);
            break;
          }
        }
      if (Ne !== wn)
        try {
          je ? S.setAttributeNS(je, $e, Ne) : S.setAttribute($e, Ne), yn(S) ? Ct(S) : rs(e.removed);
        } catch {
          T0($e, S);
        }
    }
    Yt(U.afterSanitizeAttributes, S, null);
  }, ku = function Y(S) {
    let I = null;
    const W = hi(S);
    for (Yt(U.beforeSanitizeShadowDOM, S, null); I = W.nextNode(); )
      Yt(U.uponSanitizeShadowNode, I, null), mi(I), gi(I), I.content instanceof i && Y(I.content);
    Yt(U.afterSanitizeShadowDOM, S, null);
  };
  return e.sanitize = function(Y) {
    let S = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, I = null, W = null, be = null, Re = null;
    if (vn = !Y, vn && (Y = "<!-->"), typeof Y != "string" && !di(Y))
      if (typeof Y.toString == "function") {
        if (Y = Y.toString(), typeof Y != "string")
          throw Y0("dirty is not a string, aborting");
      } else
        throw Y0("toString is not a function");
    if (!e.isSupported)
      return Y;
    if (D0 || bn(S), e.removed = [], typeof Y == "string" && (P0 = !1), P0) {
      if (Y.nodeName) {
        const Xt = ze(Y.nodeName);
        if (!N[Xt] || Ie[Xt])
          throw Y0("root node is forbidden and cannot be sanitized in-place");
      }
    } else if (Y instanceof s)
      I = ci("<!---->"), W = I.ownerDocument.importNode(Y, !0), W.nodeType === Z0.element && W.nodeName === "BODY" || W.nodeName === "HTML" ? I = W : I.appendChild(W);
    else {
      if (!Ft && !At && !gt && // eslint-disable-next-line unicorn/prefer-includes
      Y.indexOf("<") === -1)
        return w && Wt ? w.createHTML(Y) : Y;
      if (I = ci(Y), !I)
        return Ft ? null : Wt ? E : "";
    }
    I && S0 && Ct(I.firstChild);
    const $e = hi(P0 ? Y : I);
    for (; be = $e.nextNode(); )
      mi(be), gi(be), be.content instanceof i && ku(be.content);
    if (P0)
      return Y;
    if (Ft) {
      if (Vt)
        for (Re = M.call(I.ownerDocument); I.firstChild; )
          Re.appendChild(I.firstChild);
      else
        Re = I;
      return (ce.shadowroot || ce.shadowrootmode) && (Re = G.call(r, Re, !0)), Re;
    }
    let je = gt ? I.outerHTML : I.innerHTML;
    return gt && N["!doctype"] && I.ownerDocument && I.ownerDocument.doctype && I.ownerDocument.doctype.name && He(fu, I.ownerDocument.doctype.name) && (je = "<!DOCTYPE " + I.ownerDocument.doctype.name + `>
` + je), At && Or([j, oe, ee], (Xt) => {
      je = j0(je, Xt, " ");
    }), w && Wt ? w.createHTML(je) : je;
  }, e.setConfig = function() {
    let Y = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
    bn(Y), D0 = !0;
  }, e.clearConfig = function() {
    C0 = null, D0 = !1;
  }, e.isValidAttribute = function(Y, S, I) {
    C0 || bn({});
    const W = ze(Y), be = ze(S);
    return fi(W, be, I);
  }, e.addHook = function(Y, S) {
    typeof S == "function" && W0(U[Y], S);
  }, e.removeHook = function(Y, S) {
    if (S !== void 0) {
      const I = Em(U[Y], S);
      return I === -1 ? void 0 : Fm(U[Y], I, 1)[0];
    }
    return rs(U[Y]);
  }, e.removeHooks = function(Y) {
    U[Y] = [];
  }, e.removeAllHooks = function() {
    U = us();
  }, e;
}
var fS = pu();
const {
  HtmlTagHydration: pS,
  SvelteComponent: gS,
  add_render_callback: vS,
  append_hydration: _S,
  attr: bS,
  bubble: yS,
  check_outros: wS,
  children: xS,
  claim_component: kS,
  claim_element: DS,
  claim_html_tag: SS,
  claim_space: AS,
  claim_text: ES,
  create_component: FS,
  create_in_transition: CS,
  create_out_transition: TS,
  destroy_component: $S,
  detach: MS,
  element: zS,
  get_svelte_dataset: BS,
  group_outros: RS,
  init: NS,
  insert_hydration: qS,
  listen: LS,
  mount_component: IS,
  run_all: OS,
  safe_not_equal: PS,
  set_data: HS,
  space: US,
  stop_propagation: GS,
  text: VS,
  toggle_class: WS,
  transition_in: jS,
  transition_out: YS
} = window.__gradio__svelte__internal, { createEventDispatcher: XS, onMount: ZS } = window.__gradio__svelte__internal, {
  SvelteComponent: KS,
  append_hydration: QS,
  attr: JS,
  bubble: eA,
  check_outros: tA,
  children: rA,
  claim_component: nA,
  claim_element: aA,
  claim_space: iA,
  create_animation: lA,
  create_component: sA,
  destroy_component: oA,
  detach: uA,
  element: cA,
  ensure_array_like: hA,
  fix_and_outro_and_destroy_block: dA,
  fix_position: mA,
  group_outros: fA,
  init: pA,
  insert_hydration: gA,
  mount_component: vA,
  noop: _A,
  safe_not_equal: bA,
  set_style: yA,
  space: wA,
  transition_in: xA,
  transition_out: kA,
  update_keyed_each: DA
} = window.__gradio__svelte__internal, {
  SvelteComponent: SA,
  attr: AA,
  children: EA,
  claim_element: FA,
  detach: CA,
  element: TA,
  empty: $A,
  init: MA,
  insert_hydration: zA,
  noop: BA,
  safe_not_equal: RA,
  set_style: NA
} = window.__gradio__svelte__internal, {
  SvelteComponent: Vm,
  append_hydration: cs,
  assign: Wm,
  attr: o0,
  binding_callbacks: jm,
  check_outros: Ym,
  children: Xm,
  claim_component: Ka,
  claim_element: hs,
  claim_space: ds,
  claim_text: Zm,
  create_component: Qa,
  destroy_component: Ja,
  detach: Xr,
  element: ms,
  flush: Ye,
  get_spread_object: Km,
  get_spread_update: Qm,
  group_outros: Jm,
  init: e2,
  insert_hydration: xa,
  listen: fs,
  mount_component: ei,
  run_all: t2,
  safe_not_equal: r2,
  set_data: n2,
  set_input_value: ps,
  space: gs,
  text: a2,
  toggle_class: i2,
  transition_in: R0,
  transition_out: ir
} = window.__gradio__svelte__internal, { tick: l2 } = window.__gradio__svelte__internal;
function vs(n) {
  let e, t;
  const r = [
    { autoscroll: (
      /*gradio*/
      n[1].autoscroll
    ) },
    { i18n: (
      /*gradio*/
      n[1].i18n
    ) },
    /*loading_status*/
    n[10]
  ];
  let a = {};
  for (let i = 0; i < r.length; i += 1)
    a = Wm(a, r[i]);
  return e = new km({ props: a }), e.$on(
    "clear_status",
    /*clear_status_handler*/
    n[16]
  ), {
    c() {
      Qa(e.$$.fragment);
    },
    l(i) {
      Ka(e.$$.fragment, i);
    },
    m(i, l) {
      ei(e, i, l), t = !0;
    },
    p(i, l) {
      const s = l & /*gradio, loading_status*/
      1026 ? Qm(r, [
        l & /*gradio*/
        2 && { autoscroll: (
          /*gradio*/
          i[1].autoscroll
        ) },
        l & /*gradio*/
        2 && { i18n: (
          /*gradio*/
          i[1].i18n
        ) },
        l & /*loading_status*/
        1024 && Km(
          /*loading_status*/
          i[10]
        )
      ]) : {};
      e.$set(s);
    },
    i(i) {
      t || (R0(e.$$.fragment, i), t = !0);
    },
    o(i) {
      ir(e.$$.fragment, i), t = !1;
    },
    d(i) {
      Ja(e, i);
    }
  };
}
function s2(n) {
  let e;
  return {
    c() {
      e = a2(
        /*label*/
        n[2]
      );
    },
    l(t) {
      e = Zm(
        t,
        /*label*/
        n[2]
      );
    },
    m(t, r) {
      xa(t, e, r);
    },
    p(t, r) {
      r & /*label*/
      4 && n2(
        e,
        /*label*/
        t[2]
      );
    },
    d(t) {
      t && Xr(e);
    }
  };
}
function o2(n) {
  let e, t, r, a, i, l, s, u, h, d, g = (
    /*loading_status*/
    n[10] && vs(n)
  );
  return r = new pd({
    props: {
      show_label: (
        /*show_label*/
        n[7]
      ),
      info: void 0,
      $$slots: { default: [s2] },
      $$scope: { ctx: n }
    }
  }), {
    c() {
      g && g.c(), e = gs(), t = ms("label"), Qa(r.$$.fragment), a = gs(), i = ms("input"), this.h();
    },
    l(p) {
      g && g.l(p), e = ds(p), t = hs(p, "LABEL", { class: !0 });
      var v = Xm(t);
      Ka(r.$$.fragment, v), a = ds(v), i = hs(v, "INPUT", {
        "data-testid": !0,
        type: !0,
        class: !0,
        placeholder: !0,
        dir: !0
      }), v.forEach(Xr), this.h();
    },
    h() {
      o0(i, "data-testid", "textbox"), o0(i, "type", "text"), o0(i, "class", "scroll-hide svelte-2jrh70"), o0(
        i,
        "placeholder",
        /*placeholder*/
        n[6]
      ), i.disabled = l = !/*interactive*/
      n[11], o0(i, "dir", s = /*rtl*/
      n[12] ? "rtl" : "ltr"), o0(t, "class", "svelte-2jrh70"), i2(t, "container", c2);
    },
    m(p, v) {
      g && g.m(p, v), xa(p, e, v), xa(p, t, v), ei(r, t, null), cs(t, a), cs(t, i), ps(
        i,
        /*value*/
        n[0]
      ), n[18](i), u = !0, h || (d = [
        fs(
          i,
          "input",
          /*input_input_handler*/
          n[17]
        ),
        fs(
          i,
          "keypress",
          /*handle_keypress*/
          n[14]
        )
      ], h = !0);
    },
    p(p, v) {
      /*loading_status*/
      p[10] ? g ? (g.p(p, v), v & /*loading_status*/
      1024 && R0(g, 1)) : (g = vs(p), g.c(), R0(g, 1), g.m(e.parentNode, e)) : g && (Jm(), ir(g, 1, 1, () => {
        g = null;
      }), Ym());
      const k = {};
      v & /*show_label*/
      128 && (k.show_label = /*show_label*/
      p[7]), v & /*$$scope, label*/
      2097156 && (k.$$scope = { dirty: v, ctx: p }), r.$set(k), (!u || v & /*placeholder*/
      64) && o0(
        i,
        "placeholder",
        /*placeholder*/
        p[6]
      ), (!u || v & /*interactive*/
      2048 && l !== (l = !/*interactive*/
      p[11])) && (i.disabled = l), (!u || v & /*rtl*/
      4096 && s !== (s = /*rtl*/
      p[12] ? "rtl" : "ltr")) && o0(i, "dir", s), v & /*value*/
      1 && i.value !== /*value*/
      p[0] && ps(
        i,
        /*value*/
        p[0]
      );
    },
    i(p) {
      u || (R0(g), R0(r.$$.fragment, p), u = !0);
    },
    o(p) {
      ir(g), ir(r.$$.fragment, p), u = !1;
    },
    d(p) {
      p && (Xr(e), Xr(t)), g && g.d(p), Ja(r), n[18](null), h = !1, t2(d);
    }
  };
}
function u2(n) {
  let e, t;
  return e = new Hu({
    props: {
      visible: (
        /*visible*/
        n[5]
      ),
      elem_id: (
        /*elem_id*/
        n[3]
      ),
      elem_classes: (
        /*elem_classes*/
        n[4]
      ),
      scale: (
        /*scale*/
        n[8]
      ),
      min_width: (
        /*min_width*/
        n[9]
      ),
      allow_overflow: !1,
      padding: !0,
      $$slots: { default: [o2] },
      $$scope: { ctx: n }
    }
  }), {
    c() {
      Qa(e.$$.fragment);
    },
    l(r) {
      Ka(e.$$.fragment, r);
    },
    m(r, a) {
      ei(e, r, a), t = !0;
    },
    p(r, [a]) {
      const i = {};
      a & /*visible*/
      32 && (i.visible = /*visible*/
      r[5]), a & /*elem_id*/
      8 && (i.elem_id = /*elem_id*/
      r[3]), a & /*elem_classes*/
      16 && (i.elem_classes = /*elem_classes*/
      r[4]), a & /*scale*/
      256 && (i.scale = /*scale*/
      r[8]), a & /*min_width*/
      512 && (i.min_width = /*min_width*/
      r[9]), a & /*$$scope, placeholder, interactive, rtl, value, el, show_label, label, gradio, loading_status*/
      2112711 && (i.$$scope = { dirty: a, ctx: r }), e.$set(i);
    },
    i(r) {
      t || (R0(e.$$.fragment, r), t = !0);
    },
    o(r) {
      ir(e.$$.fragment, r), t = !1;
    },
    d(r) {
      Ja(e, r);
    }
  };
}
const c2 = !0;
function h2(n, e, t) {
  var r = this && this.__awaiter || function(M, B, G, U) {
    function j(oe) {
      return oe instanceof G ? oe : new G(function(ee) {
        ee(oe);
      });
    }
    return new (G || (G = Promise))(function(oe, ee) {
      function ue(ne) {
        try {
          Ee(U.next(ne));
        } catch (ve) {
          ee(ve);
        }
      }
      function fe(ne) {
        try {
          Ee(U.throw(ne));
        } catch (ve) {
          ee(ve);
        }
      }
      function Ee(ne) {
        ne.done ? oe(ne.value) : j(ne.value).then(ue, fe);
      }
      Ee((U = U.apply(M, B || [])).next());
    });
  };
  let { gradio: a } = e, { label: i = "Textbox" } = e, { elem_id: l = "" } = e, { elem_classes: s = [] } = e, { visible: u = !0 } = e, { value: h = "" } = e, { placeholder: d = "" } = e, { show_label: g } = e, { scale: p = null } = e, { min_width: v = void 0 } = e, { loading_status: k = void 0 } = e, { value_is_output: A = !1 } = e, { interactive: C } = e, { rtl: z = !1 } = e, x;
  function _() {
    a.dispatch("change"), A || a.dispatch("input");
  }
  function w(M) {
    return r(this, void 0, void 0, function* () {
      yield l2(), M.key === "Enter" && (M.preventDefault(), a.dispatch("submit"));
    });
  }
  const E = () => a.dispatch("clear_status", k);
  function T() {
    h = this.value, t(0, h);
  }
  function $(M) {
    jm[M ? "unshift" : "push"](() => {
      x = M, t(13, x);
    });
  }
  return n.$$set = (M) => {
    "gradio" in M && t(1, a = M.gradio), "label" in M && t(2, i = M.label), "elem_id" in M && t(3, l = M.elem_id), "elem_classes" in M && t(4, s = M.elem_classes), "visible" in M && t(5, u = M.visible), "value" in M && t(0, h = M.value), "placeholder" in M && t(6, d = M.placeholder), "show_label" in M && t(7, g = M.show_label), "scale" in M && t(8, p = M.scale), "min_width" in M && t(9, v = M.min_width), "loading_status" in M && t(10, k = M.loading_status), "value_is_output" in M && t(15, A = M.value_is_output), "interactive" in M && t(11, C = M.interactive), "rtl" in M && t(12, z = M.rtl);
  }, n.$$.update = () => {
    n.$$.dirty & /*value*/
    1 && h === null && t(0, h = ""), n.$$.dirty & /*value*/
    1 && _();
  }, [
    h,
    a,
    i,
    l,
    s,
    u,
    d,
    g,
    p,
    v,
    k,
    C,
    z,
    x,
    w,
    A,
    E,
    T,
    $
  ];
}
class qA extends Vm {
  constructor(e) {
    super(), e2(this, e, h2, u2, r2, {
      gradio: 1,
      label: 2,
      elem_id: 3,
      elem_classes: 4,
      visible: 5,
      value: 0,
      placeholder: 6,
      show_label: 7,
      scale: 8,
      min_width: 9,
      loading_status: 10,
      value_is_output: 15,
      interactive: 11,
      rtl: 12
    });
  }
  get gradio() {
    return this.$$.ctx[1];
  }
  set gradio(e) {
    this.$$set({ gradio: e }), Ye();
  }
  get label() {
    return this.$$.ctx[2];
  }
  set label(e) {
    this.$$set({ label: e }), Ye();
  }
  get elem_id() {
    return this.$$.ctx[3];
  }
  set elem_id(e) {
    this.$$set({ elem_id: e }), Ye();
  }
  get elem_classes() {
    return this.$$.ctx[4];
  }
  set elem_classes(e) {
    this.$$set({ elem_classes: e }), Ye();
  }
  get visible() {
    return this.$$.ctx[5];
  }
  set visible(e) {
    this.$$set({ visible: e }), Ye();
  }
  get value() {
    return this.$$.ctx[0];
  }
  set value(e) {
    this.$$set({ value: e }), Ye();
  }
  get placeholder() {
    return this.$$.ctx[6];
  }
  set placeholder(e) {
    this.$$set({ placeholder: e }), Ye();
  }
  get show_label() {
    return this.$$.ctx[7];
  }
  set show_label(e) {
    this.$$set({ show_label: e }), Ye();
  }
  get scale() {
    return this.$$.ctx[8];
  }
  set scale(e) {
    this.$$set({ scale: e }), Ye();
  }
  get min_width() {
    return this.$$.ctx[9];
  }
  set min_width(e) {
    this.$$set({ min_width: e }), Ye();
  }
  get loading_status() {
    return this.$$.ctx[10];
  }
  set loading_status(e) {
    this.$$set({ loading_status: e }), Ye();
  }
  get value_is_output() {
    return this.$$.ctx[15];
  }
  set value_is_output(e) {
    this.$$set({ value_is_output: e }), Ye();
  }
  get interactive() {
    return this.$$.ctx[11];
  }
  set interactive(e) {
    this.$$set({ interactive: e }), Ye();
  }
  get rtl() {
    return this.$$.ctx[12];
  }
  set rtl(e) {
    this.$$set({ rtl: e }), Ye();
  }
}
export {
  qA as I,
  fl as c,
  f2 as g,
  m2 as k,
  fS as p
};
