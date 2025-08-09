var Vp = Object.defineProperty;
var Zp = (e, t, r) => t in e ? Vp(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r;
var ot = (e, t, r) => Zp(e, typeof t != "symbol" ? t + "" : t, r);
import { g as Kp, c as Qp, p as pr } from "./Index-BG2POTv1.js";
var Rl = { exports: {} };
(function(e, t) {
  (function(r, i) {
    e.exports = i();
  })(Qp, function() {
    var r = 1e3, i = 6e4, a = 36e5, n = "millisecond", o = "second", s = "minute", l = "hour", c = "day", h = "week", u = "month", f = "quarter", d = "year", g = "date", m = "Invalid Date", y = /^(\d{4})[-/]?(\d{1,2})?[-/]?(\d{0,2})[Tt\s]*(\d{1,2})?:?(\d{1,2})?:?(\d{1,2})?[.:]?(\d+)?$/, x = /\[([^\]]+)]|Y{1,4}|M{1,4}|D{1,2}|d{1,4}|H{1,2}|h{1,2}|a|A|m{1,2}|s{1,2}|Z{1,2}|SSS/g, b = { name: "en", weekdays: "Sunday_Monday_Tuesday_Wednesday_Thursday_Friday_Saturday".split("_"), months: "January_February_March_April_May_June_July_August_September_October_November_December".split("_"), ordinal: function(B) {
      var M = ["th", "st", "nd", "rd"], T = B % 100;
      return "[" + B + (M[(T - 20) % 10] || M[T] || M[0]) + "]";
    } }, k = function(B, M, T) {
      var A = String(B);
      return !A || A.length >= M ? B : "" + Array(M + 1 - A.length).join(T) + B;
    }, S = { s: k, z: function(B) {
      var M = -B.utcOffset(), T = Math.abs(M), A = Math.floor(T / 60), L = T % 60;
      return (M <= 0 ? "+" : "-") + k(A, 2, "0") + ":" + k(L, 2, "0");
    }, m: function B(M, T) {
      if (M.date() < T.date()) return -B(T, M);
      var A = 12 * (T.year() - M.year()) + (T.month() - M.month()), L = M.clone().add(A, u), N = T - L < 0, U = M.clone().add(A + (N ? -1 : 1), u);
      return +(-(A + (T - L) / (N ? L - U : U - L)) || 0);
    }, a: function(B) {
      return B < 0 ? Math.ceil(B) || 0 : Math.floor(B);
    }, p: function(B) {
      return { M: u, y: d, w: h, d: c, D: g, h: l, m: s, s: o, ms: n, Q: f }[B] || String(B || "").toLowerCase().replace(/s$/, "");
    }, u: function(B) {
      return B === void 0;
    } }, w = "en", C = {};
    C[w] = b;
    var _ = "$isDayjsObject", E = function(B) {
      return B instanceof I || !(!B || !B[_]);
    }, R = function B(M, T, A) {
      var L;
      if (!M) return w;
      if (typeof M == "string") {
        var N = M.toLowerCase();
        C[N] && (L = N), T && (C[N] = T, L = N);
        var U = M.split("-");
        if (!L && U.length > 1) return B(U[0]);
      } else {
        var J = M.name;
        C[J] = M, L = J;
      }
      return !A && L && (w = L), L || !A && w;
    }, O = function(B, M) {
      if (E(B)) return B.clone();
      var T = typeof M == "object" ? M : {};
      return T.date = B, T.args = arguments, new I(T);
    }, $ = S;
    $.l = R, $.i = E, $.w = function(B, M) {
      return O(B, { locale: M.$L, utc: M.$u, x: M.$x, $offset: M.$offset });
    };
    var I = function() {
      function B(T) {
        this.$L = R(T.locale, null, !0), this.parse(T), this.$x = this.$x || T.x || {}, this[_] = !0;
      }
      var M = B.prototype;
      return M.parse = function(T) {
        this.$d = function(A) {
          var L = A.date, N = A.utc;
          if (L === null) return /* @__PURE__ */ new Date(NaN);
          if ($.u(L)) return /* @__PURE__ */ new Date();
          if (L instanceof Date) return new Date(L);
          if (typeof L == "string" && !/Z$/i.test(L)) {
            var U = L.match(y);
            if (U) {
              var J = U[2] - 1 || 0, lt = (U[7] || "0").substring(0, 3);
              return N ? new Date(Date.UTC(U[1], J, U[3] || 1, U[4] || 0, U[5] || 0, U[6] || 0, lt)) : new Date(U[1], J, U[3] || 1, U[4] || 0, U[5] || 0, U[6] || 0, lt);
            }
          }
          return new Date(L);
        }(T), this.init();
      }, M.init = function() {
        var T = this.$d;
        this.$y = T.getFullYear(), this.$M = T.getMonth(), this.$D = T.getDate(), this.$W = T.getDay(), this.$H = T.getHours(), this.$m = T.getMinutes(), this.$s = T.getSeconds(), this.$ms = T.getMilliseconds();
      }, M.$utils = function() {
        return $;
      }, M.isValid = function() {
        return this.$d.toString() !== m;
      }, M.isSame = function(T, A) {
        var L = O(T);
        return this.startOf(A) <= L && L <= this.endOf(A);
      }, M.isAfter = function(T, A) {
        return O(T) < this.startOf(A);
      }, M.isBefore = function(T, A) {
        return this.endOf(A) < O(T);
      }, M.$g = function(T, A, L) {
        return $.u(T) ? this[A] : this.set(L, T);
      }, M.unix = function() {
        return Math.floor(this.valueOf() / 1e3);
      }, M.valueOf = function() {
        return this.$d.getTime();
      }, M.startOf = function(T, A) {
        var L = this, N = !!$.u(A) || A, U = $.p(T), J = function(dt, ut) {
          var kt = $.w(L.$u ? Date.UTC(L.$y, ut, dt) : new Date(L.$y, ut, dt), L);
          return N ? kt : kt.endOf(c);
        }, lt = function(dt, ut) {
          return $.w(L.toDate()[dt].apply(L.toDate("s"), (N ? [0, 0, 0, 0] : [23, 59, 59, 999]).slice(ut)), L);
        }, ft = this.$W, yt = this.$M, Mt = this.$D, tt = "set" + (this.$u ? "UTC" : "");
        switch (U) {
          case d:
            return N ? J(1, 0) : J(31, 11);
          case u:
            return N ? J(1, yt) : J(0, yt + 1);
          case h:
            var it = this.$locale().weekStart || 0, gt = (ft < it ? ft + 7 : ft) - it;
            return J(N ? Mt - gt : Mt + (6 - gt), yt);
          case c:
          case g:
            return lt(tt + "Hours", 0);
          case l:
            return lt(tt + "Minutes", 1);
          case s:
            return lt(tt + "Seconds", 2);
          case o:
            return lt(tt + "Milliseconds", 3);
          default:
            return this.clone();
        }
      }, M.endOf = function(T) {
        return this.startOf(T, !1);
      }, M.$set = function(T, A) {
        var L, N = $.p(T), U = "set" + (this.$u ? "UTC" : ""), J = (L = {}, L[c] = U + "Date", L[g] = U + "Date", L[u] = U + "Month", L[d] = U + "FullYear", L[l] = U + "Hours", L[s] = U + "Minutes", L[o] = U + "Seconds", L[n] = U + "Milliseconds", L)[N], lt = N === c ? this.$D + (A - this.$W) : A;
        if (N === u || N === d) {
          var ft = this.clone().set(g, 1);
          ft.$d[J](lt), ft.init(), this.$d = ft.set(g, Math.min(this.$D, ft.daysInMonth())).$d;
        } else J && this.$d[J](lt);
        return this.init(), this;
      }, M.set = function(T, A) {
        return this.clone().$set(T, A);
      }, M.get = function(T) {
        return this[$.p(T)]();
      }, M.add = function(T, A) {
        var L, N = this;
        T = Number(T);
        var U = $.p(A), J = function(yt) {
          var Mt = O(N);
          return $.w(Mt.date(Mt.date() + Math.round(yt * T)), N);
        };
        if (U === u) return this.set(u, this.$M + T);
        if (U === d) return this.set(d, this.$y + T);
        if (U === c) return J(1);
        if (U === h) return J(7);
        var lt = (L = {}, L[s] = i, L[l] = a, L[o] = r, L)[U] || 1, ft = this.$d.getTime() + T * lt;
        return $.w(ft, this);
      }, M.subtract = function(T, A) {
        return this.add(-1 * T, A);
      }, M.format = function(T) {
        var A = this, L = this.$locale();
        if (!this.isValid()) return L.invalidDate || m;
        var N = T || "YYYY-MM-DDTHH:mm:ssZ", U = $.z(this), J = this.$H, lt = this.$m, ft = this.$M, yt = L.weekdays, Mt = L.months, tt = L.meridiem, it = function(ut, kt, xe, $r) {
          return ut && (ut[kt] || ut(A, N)) || xe[kt].slice(0, $r);
        }, gt = function(ut) {
          return $.s(J % 12 || 12, ut, "0");
        }, dt = tt || function(ut, kt, xe) {
          var $r = ut < 12 ? "AM" : "PM";
          return xe ? $r.toLowerCase() : $r;
        };
        return N.replace(x, function(ut, kt) {
          return kt || function(xe) {
            switch (xe) {
              case "YY":
                return String(A.$y).slice(-2);
              case "YYYY":
                return $.s(A.$y, 4, "0");
              case "M":
                return ft + 1;
              case "MM":
                return $.s(ft + 1, 2, "0");
              case "MMM":
                return it(L.monthsShort, ft, Mt, 3);
              case "MMMM":
                return it(Mt, ft);
              case "D":
                return A.$D;
              case "DD":
                return $.s(A.$D, 2, "0");
              case "d":
                return String(A.$W);
              case "dd":
                return it(L.weekdaysMin, A.$W, yt, 2);
              case "ddd":
                return it(L.weekdaysShort, A.$W, yt, 3);
              case "dddd":
                return yt[A.$W];
              case "H":
                return String(J);
              case "HH":
                return $.s(J, 2, "0");
              case "h":
                return gt(1);
              case "hh":
                return gt(2);
              case "a":
                return dt(J, lt, !0);
              case "A":
                return dt(J, lt, !1);
              case "m":
                return String(lt);
              case "mm":
                return $.s(lt, 2, "0");
              case "s":
                return String(A.$s);
              case "ss":
                return $.s(A.$s, 2, "0");
              case "SSS":
                return $.s(A.$ms, 3, "0");
              case "Z":
                return U;
            }
            return null;
          }(ut) || U.replace(":", "");
        });
      }, M.utcOffset = function() {
        return 15 * -Math.round(this.$d.getTimezoneOffset() / 15);
      }, M.diff = function(T, A, L) {
        var N, U = this, J = $.p(A), lt = O(T), ft = (lt.utcOffset() - this.utcOffset()) * i, yt = this - lt, Mt = function() {
          return $.m(U, lt);
        };
        switch (J) {
          case d:
            N = Mt() / 12;
            break;
          case u:
            N = Mt();
            break;
          case f:
            N = Mt() / 3;
            break;
          case h:
            N = (yt - ft) / 6048e5;
            break;
          case c:
            N = (yt - ft) / 864e5;
            break;
          case l:
            N = yt / a;
            break;
          case s:
            N = yt / i;
            break;
          case o:
            N = yt / r;
            break;
          default:
            N = yt;
        }
        return L ? N : $.a(N);
      }, M.daysInMonth = function() {
        return this.endOf(u).$D;
      }, M.$locale = function() {
        return C[this.$L];
      }, M.locale = function(T, A) {
        if (!T) return this.$L;
        var L = this.clone(), N = R(T, A, !0);
        return N && (L.$L = N), L;
      }, M.clone = function() {
        return $.w(this.$d, this);
      }, M.toDate = function() {
        return new Date(this.valueOf());
      }, M.toJSON = function() {
        return this.isValid() ? this.toISOString() : null;
      }, M.toISOString = function() {
        return this.$d.toISOString();
      }, M.toString = function() {
        return this.$d.toUTCString();
      }, B;
    }(), D = I.prototype;
    return O.prototype = D, [["$ms", n], ["$s", o], ["$m", s], ["$H", l], ["$W", c], ["$M", u], ["$y", d], ["$D", g]].forEach(function(B) {
      D[B[1]] = function(M) {
        return this.$g(M, B[0], B[1]);
      };
    }), O.extend = function(B, M) {
      return B.$i || (B(M, I, O), B.$i = !0), O;
    }, O.locale = R, O.isDayjs = E, O.unix = function(B) {
      return O(1e3 * B);
    }, O.en = C[w], O.Ls = C, O.p = {}, O;
  });
})(Rl);
var Jp = Rl.exports;
const tg = /* @__PURE__ */ Kp(Jp), Mi = {
  /* CLAMP */
  min: {
    r: 0,
    g: 0,
    b: 0,
    s: 0,
    l: 0,
    a: 0
  },
  max: {
    r: 255,
    g: 255,
    b: 255,
    h: 360,
    s: 100,
    l: 100,
    a: 1
  },
  clamp: {
    r: (e) => e >= 255 ? 255 : e < 0 ? 0 : e,
    g: (e) => e >= 255 ? 255 : e < 0 ? 0 : e,
    b: (e) => e >= 255 ? 255 : e < 0 ? 0 : e,
    h: (e) => e % 360,
    s: (e) => e >= 100 ? 100 : e < 0 ? 0 : e,
    l: (e) => e >= 100 ? 100 : e < 0 ? 0 : e,
    a: (e) => e >= 1 ? 1 : e < 0 ? 0 : e
  },
  /* CONVERSION */
  //SOURCE: https://planetcalc.com/7779
  toLinear: (e) => {
    const t = e / 255;
    return e > 0.03928 ? Math.pow((t + 0.055) / 1.055, 2.4) : t / 12.92;
  },
  //SOURCE: https://gist.github.com/mjackson/5311256
  hue2rgb: (e, t, r) => (r < 0 && (r += 1), r > 1 && (r -= 1), r < 1 / 6 ? e + (t - e) * 6 * r : r < 1 / 2 ? t : r < 2 / 3 ? e + (t - e) * (2 / 3 - r) * 6 : e),
  hsl2rgb: ({ h: e, s: t, l: r }, i) => {
    if (!t)
      return r * 2.55;
    e /= 360, t /= 100, r /= 100;
    const a = r < 0.5 ? r * (1 + t) : r + t - r * t, n = 2 * r - a;
    switch (i) {
      case "r":
        return Mi.hue2rgb(n, a, e + 1 / 3) * 255;
      case "g":
        return Mi.hue2rgb(n, a, e) * 255;
      case "b":
        return Mi.hue2rgb(n, a, e - 1 / 3) * 255;
    }
  },
  rgb2hsl: ({ r: e, g: t, b: r }, i) => {
    e /= 255, t /= 255, r /= 255;
    const a = Math.max(e, t, r), n = Math.min(e, t, r), o = (a + n) / 2;
    if (i === "l")
      return o * 100;
    if (a === n)
      return 0;
    const s = a - n, l = o > 0.5 ? s / (2 - a - n) : s / (a + n);
    if (i === "s")
      return l * 100;
    switch (a) {
      case e:
        return ((t - r) / s + (t < r ? 6 : 0)) * 60;
      case t:
        return ((r - e) / s + 2) * 60;
      case r:
        return ((e - t) / s + 4) * 60;
      default:
        return -1;
    }
  }
}, eg = {
  /* API */
  clamp: (e, t, r) => t > r ? Math.min(t, Math.max(r, e)) : Math.min(r, Math.max(t, e)),
  round: (e) => Math.round(e * 1e10) / 1e10
}, rg = {
  /* API */
  dec2hex: (e) => {
    const t = Math.round(e).toString(16);
    return t.length > 1 ? t : `0${t}`;
  }
}, K = {
  channel: Mi,
  lang: eg,
  unit: rg
}, be = {};
for (let e = 0; e <= 255; e++)
  be[e] = K.unit.dec2hex(e);
const St = {
  ALL: 0,
  RGB: 1,
  HSL: 2
};
class ig {
  constructor() {
    this.type = St.ALL;
  }
  /* API */
  get() {
    return this.type;
  }
  set(t) {
    if (this.type && this.type !== t)
      throw new Error("Cannot change both RGB and HSL channels at the same time");
    this.type = t;
  }
  reset() {
    this.type = St.ALL;
  }
  is(t) {
    return this.type === t;
  }
}
class ag {
  /* CONSTRUCTOR */
  constructor(t, r) {
    this.color = r, this.changed = !1, this.data = t, this.type = new ig();
  }
  /* API */
  set(t, r) {
    return this.color = r, this.changed = !1, this.data = t, this.type.type = St.ALL, this;
  }
  /* HELPERS */
  _ensureHSL() {
    const t = this.data, { h: r, s: i, l: a } = t;
    r === void 0 && (t.h = K.channel.rgb2hsl(t, "h")), i === void 0 && (t.s = K.channel.rgb2hsl(t, "s")), a === void 0 && (t.l = K.channel.rgb2hsl(t, "l"));
  }
  _ensureRGB() {
    const t = this.data, { r, g: i, b: a } = t;
    r === void 0 && (t.r = K.channel.hsl2rgb(t, "r")), i === void 0 && (t.g = K.channel.hsl2rgb(t, "g")), a === void 0 && (t.b = K.channel.hsl2rgb(t, "b"));
  }
  /* GETTERS */
  get r() {
    const t = this.data, r = t.r;
    return !this.type.is(St.HSL) && r !== void 0 ? r : (this._ensureHSL(), K.channel.hsl2rgb(t, "r"));
  }
  get g() {
    const t = this.data, r = t.g;
    return !this.type.is(St.HSL) && r !== void 0 ? r : (this._ensureHSL(), K.channel.hsl2rgb(t, "g"));
  }
  get b() {
    const t = this.data, r = t.b;
    return !this.type.is(St.HSL) && r !== void 0 ? r : (this._ensureHSL(), K.channel.hsl2rgb(t, "b"));
  }
  get h() {
    const t = this.data, r = t.h;
    return !this.type.is(St.RGB) && r !== void 0 ? r : (this._ensureRGB(), K.channel.rgb2hsl(t, "h"));
  }
  get s() {
    const t = this.data, r = t.s;
    return !this.type.is(St.RGB) && r !== void 0 ? r : (this._ensureRGB(), K.channel.rgb2hsl(t, "s"));
  }
  get l() {
    const t = this.data, r = t.l;
    return !this.type.is(St.RGB) && r !== void 0 ? r : (this._ensureRGB(), K.channel.rgb2hsl(t, "l"));
  }
  get a() {
    return this.data.a;
  }
  /* SETTERS */
  set r(t) {
    this.type.set(St.RGB), this.changed = !0, this.data.r = t;
  }
  set g(t) {
    this.type.set(St.RGB), this.changed = !0, this.data.g = t;
  }
  set b(t) {
    this.type.set(St.RGB), this.changed = !0, this.data.b = t;
  }
  set h(t) {
    this.type.set(St.HSL), this.changed = !0, this.data.h = t;
  }
  set s(t) {
    this.type.set(St.HSL), this.changed = !0, this.data.s = t;
  }
  set l(t) {
    this.type.set(St.HSL), this.changed = !0, this.data.l = t;
  }
  set a(t) {
    this.changed = !0, this.data.a = t;
  }
}
const va = new ag({ r: 0, g: 0, b: 0, a: 0 }, "transparent"), ir = {
  /* VARIABLES */
  re: /^#((?:[a-f0-9]{2}){2,4}|[a-f0-9]{3})$/i,
  /* API */
  parse: (e) => {
    if (e.charCodeAt(0) !== 35)
      return;
    const t = e.match(ir.re);
    if (!t)
      return;
    const r = t[1], i = parseInt(r, 16), a = r.length, n = a % 4 === 0, o = a > 4, s = o ? 1 : 17, l = o ? 8 : 4, c = n ? 0 : -1, h = o ? 255 : 15;
    return va.set({
      r: (i >> l * (c + 3) & h) * s,
      g: (i >> l * (c + 2) & h) * s,
      b: (i >> l * (c + 1) & h) * s,
      a: n ? (i & h) * s / 255 : 1
    }, e);
  },
  stringify: (e) => {
    const { r: t, g: r, b: i, a } = e;
    return a < 1 ? `#${be[Math.round(t)]}${be[Math.round(r)]}${be[Math.round(i)]}${be[Math.round(a * 255)]}` : `#${be[Math.round(t)]}${be[Math.round(r)]}${be[Math.round(i)]}`;
  }
}, Ee = {
  /* VARIABLES */
  re: /^hsla?\(\s*?(-?(?:\d+(?:\.\d+)?|(?:\.\d+))(?:e-?\d+)?(?:deg|grad|rad|turn)?)\s*?(?:,|\s)\s*?(-?(?:\d+(?:\.\d+)?|(?:\.\d+))(?:e-?\d+)?%)\s*?(?:,|\s)\s*?(-?(?:\d+(?:\.\d+)?|(?:\.\d+))(?:e-?\d+)?%)(?:\s*?(?:,|\/)\s*?\+?(-?(?:\d+(?:\.\d+)?|(?:\.\d+))(?:e-?\d+)?(%)?))?\s*?\)$/i,
  hueRe: /^(.+?)(deg|grad|rad|turn)$/i,
  /* HELPERS */
  _hue2deg: (e) => {
    const t = e.match(Ee.hueRe);
    if (t) {
      const [, r, i] = t;
      switch (i) {
        case "grad":
          return K.channel.clamp.h(parseFloat(r) * 0.9);
        case "rad":
          return K.channel.clamp.h(parseFloat(r) * 180 / Math.PI);
        case "turn":
          return K.channel.clamp.h(parseFloat(r) * 360);
      }
    }
    return K.channel.clamp.h(parseFloat(e));
  },
  /* API */
  parse: (e) => {
    const t = e.charCodeAt(0);
    if (t !== 104 && t !== 72)
      return;
    const r = e.match(Ee.re);
    if (!r)
      return;
    const [, i, a, n, o, s] = r;
    return va.set({
      h: Ee._hue2deg(i),
      s: K.channel.clamp.s(parseFloat(a)),
      l: K.channel.clamp.l(parseFloat(n)),
      a: o ? K.channel.clamp.a(s ? parseFloat(o) / 100 : parseFloat(o)) : 1
    }, e);
  },
  stringify: (e) => {
    const { h: t, s: r, l: i, a } = e;
    return a < 1 ? `hsla(${K.lang.round(t)}, ${K.lang.round(r)}%, ${K.lang.round(i)}%, ${a})` : `hsl(${K.lang.round(t)}, ${K.lang.round(r)}%, ${K.lang.round(i)}%)`;
  }
}, Yr = {
  /* VARIABLES */
  colors: {
    aliceblue: "#f0f8ff",
    antiquewhite: "#faebd7",
    aqua: "#00ffff",
    aquamarine: "#7fffd4",
    azure: "#f0ffff",
    beige: "#f5f5dc",
    bisque: "#ffe4c4",
    black: "#000000",
    blanchedalmond: "#ffebcd",
    blue: "#0000ff",
    blueviolet: "#8a2be2",
    brown: "#a52a2a",
    burlywood: "#deb887",
    cadetblue: "#5f9ea0",
    chartreuse: "#7fff00",
    chocolate: "#d2691e",
    coral: "#ff7f50",
    cornflowerblue: "#6495ed",
    cornsilk: "#fff8dc",
    crimson: "#dc143c",
    cyanaqua: "#00ffff",
    darkblue: "#00008b",
    darkcyan: "#008b8b",
    darkgoldenrod: "#b8860b",
    darkgray: "#a9a9a9",
    darkgreen: "#006400",
    darkgrey: "#a9a9a9",
    darkkhaki: "#bdb76b",
    darkmagenta: "#8b008b",
    darkolivegreen: "#556b2f",
    darkorange: "#ff8c00",
    darkorchid: "#9932cc",
    darkred: "#8b0000",
    darksalmon: "#e9967a",
    darkseagreen: "#8fbc8f",
    darkslateblue: "#483d8b",
    darkslategray: "#2f4f4f",
    darkslategrey: "#2f4f4f",
    darkturquoise: "#00ced1",
    darkviolet: "#9400d3",
    deeppink: "#ff1493",
    deepskyblue: "#00bfff",
    dimgray: "#696969",
    dimgrey: "#696969",
    dodgerblue: "#1e90ff",
    firebrick: "#b22222",
    floralwhite: "#fffaf0",
    forestgreen: "#228b22",
    fuchsia: "#ff00ff",
    gainsboro: "#dcdcdc",
    ghostwhite: "#f8f8ff",
    gold: "#ffd700",
    goldenrod: "#daa520",
    gray: "#808080",
    green: "#008000",
    greenyellow: "#adff2f",
    grey: "#808080",
    honeydew: "#f0fff0",
    hotpink: "#ff69b4",
    indianred: "#cd5c5c",
    indigo: "#4b0082",
    ivory: "#fffff0",
    khaki: "#f0e68c",
    lavender: "#e6e6fa",
    lavenderblush: "#fff0f5",
    lawngreen: "#7cfc00",
    lemonchiffon: "#fffacd",
    lightblue: "#add8e6",
    lightcoral: "#f08080",
    lightcyan: "#e0ffff",
    lightgoldenrodyellow: "#fafad2",
    lightgray: "#d3d3d3",
    lightgreen: "#90ee90",
    lightgrey: "#d3d3d3",
    lightpink: "#ffb6c1",
    lightsalmon: "#ffa07a",
    lightseagreen: "#20b2aa",
    lightskyblue: "#87cefa",
    lightslategray: "#778899",
    lightslategrey: "#778899",
    lightsteelblue: "#b0c4de",
    lightyellow: "#ffffe0",
    lime: "#00ff00",
    limegreen: "#32cd32",
    linen: "#faf0e6",
    magenta: "#ff00ff",
    maroon: "#800000",
    mediumaquamarine: "#66cdaa",
    mediumblue: "#0000cd",
    mediumorchid: "#ba55d3",
    mediumpurple: "#9370db",
    mediumseagreen: "#3cb371",
    mediumslateblue: "#7b68ee",
    mediumspringgreen: "#00fa9a",
    mediumturquoise: "#48d1cc",
    mediumvioletred: "#c71585",
    midnightblue: "#191970",
    mintcream: "#f5fffa",
    mistyrose: "#ffe4e1",
    moccasin: "#ffe4b5",
    navajowhite: "#ffdead",
    navy: "#000080",
    oldlace: "#fdf5e6",
    olive: "#808000",
    olivedrab: "#6b8e23",
    orange: "#ffa500",
    orangered: "#ff4500",
    orchid: "#da70d6",
    palegoldenrod: "#eee8aa",
    palegreen: "#98fb98",
    paleturquoise: "#afeeee",
    palevioletred: "#db7093",
    papayawhip: "#ffefd5",
    peachpuff: "#ffdab9",
    peru: "#cd853f",
    pink: "#ffc0cb",
    plum: "#dda0dd",
    powderblue: "#b0e0e6",
    purple: "#800080",
    rebeccapurple: "#663399",
    red: "#ff0000",
    rosybrown: "#bc8f8f",
    royalblue: "#4169e1",
    saddlebrown: "#8b4513",
    salmon: "#fa8072",
    sandybrown: "#f4a460",
    seagreen: "#2e8b57",
    seashell: "#fff5ee",
    sienna: "#a0522d",
    silver: "#c0c0c0",
    skyblue: "#87ceeb",
    slateblue: "#6a5acd",
    slategray: "#708090",
    slategrey: "#708090",
    snow: "#fffafa",
    springgreen: "#00ff7f",
    tan: "#d2b48c",
    teal: "#008080",
    thistle: "#d8bfd8",
    transparent: "#00000000",
    turquoise: "#40e0d0",
    violet: "#ee82ee",
    wheat: "#f5deb3",
    white: "#ffffff",
    whitesmoke: "#f5f5f5",
    yellow: "#ffff00",
    yellowgreen: "#9acd32"
  },
  /* API */
  parse: (e) => {
    e = e.toLowerCase();
    const t = Yr.colors[e];
    if (t)
      return ir.parse(t);
  },
  stringify: (e) => {
    const t = ir.stringify(e);
    for (const r in Yr.colors)
      if (Yr.colors[r] === t)
        return r;
  }
}, Ir = {
  /* VARIABLES */
  re: /^rgba?\(\s*?(-?(?:\d+(?:\.\d+)?|(?:\.\d+))(?:e\d+)?(%?))\s*?(?:,|\s)\s*?(-?(?:\d+(?:\.\d+)?|(?:\.\d+))(?:e\d+)?(%?))\s*?(?:,|\s)\s*?(-?(?:\d+(?:\.\d+)?|(?:\.\d+))(?:e\d+)?(%?))(?:\s*?(?:,|\/)\s*?\+?(-?(?:\d+(?:\.\d+)?|(?:\.\d+))(?:e\d+)?(%?)))?\s*?\)$/i,
  /* API */
  parse: (e) => {
    const t = e.charCodeAt(0);
    if (t !== 114 && t !== 82)
      return;
    const r = e.match(Ir.re);
    if (!r)
      return;
    const [, i, a, n, o, s, l, c, h] = r;
    return va.set({
      r: K.channel.clamp.r(a ? parseFloat(i) * 2.55 : parseFloat(i)),
      g: K.channel.clamp.g(o ? parseFloat(n) * 2.55 : parseFloat(n)),
      b: K.channel.clamp.b(l ? parseFloat(s) * 2.55 : parseFloat(s)),
      a: c ? K.channel.clamp.a(h ? parseFloat(c) / 100 : parseFloat(c)) : 1
    }, e);
  },
  stringify: (e) => {
    const { r: t, g: r, b: i, a } = e;
    return a < 1 ? `rgba(${K.lang.round(t)}, ${K.lang.round(r)}, ${K.lang.round(i)}, ${K.lang.round(a)})` : `rgb(${K.lang.round(t)}, ${K.lang.round(r)}, ${K.lang.round(i)})`;
  }
}, ee = {
  /* VARIABLES */
  format: {
    keyword: Yr,
    hex: ir,
    rgb: Ir,
    rgba: Ir,
    hsl: Ee,
    hsla: Ee
  },
  /* API */
  parse: (e) => {
    if (typeof e != "string")
      return e;
    const t = ir.parse(e) || Ir.parse(e) || Ee.parse(e) || Yr.parse(e);
    if (t)
      return t;
    throw new Error(`Unsupported color format: "${e}"`);
  },
  stringify: (e) => !e.changed && e.color ? e.color : e.type.is(St.HSL) || e.data.r === void 0 ? Ee.stringify(e) : e.a < 1 || !Number.isInteger(e.r) || !Number.isInteger(e.g) || !Number.isInteger(e.b) ? Ir.stringify(e) : ir.stringify(e)
}, Pl = (e, t) => {
  const r = ee.parse(e);
  for (const i in t)
    r[i] = K.channel.clamp[i](t[i]);
  return ee.stringify(r);
}, Gr = (e, t, r = 0, i = 1) => {
  if (typeof e != "number")
    return Pl(e, { a: t });
  const a = va.set({
    r: K.channel.clamp.r(e),
    g: K.channel.clamp.g(t),
    b: K.channel.clamp.b(r),
    a: K.channel.clamp.a(i)
  });
  return ee.stringify(a);
}, ng = (e) => {
  const { r: t, g: r, b: i } = ee.parse(e), a = 0.2126 * K.channel.toLinear(t) + 0.7152 * K.channel.toLinear(r) + 0.0722 * K.channel.toLinear(i);
  return K.lang.round(a);
}, sg = (e) => ng(e) >= 0.5, ci = (e) => !sg(e), Il = (e, t, r) => {
  const i = ee.parse(e), a = i[t], n = K.channel.clamp[t](a + r);
  return a !== n && (i[t] = n), ee.stringify(i);
}, z = (e, t) => Il(e, "l", t), X = (e, t) => Il(e, "l", -t), v = (e, t) => {
  const r = ee.parse(e), i = {};
  for (const a in t)
    t[a] && (i[a] = r[a] + t[a]);
  return Pl(e, i);
}, og = (e, t, r = 50) => {
  const { r: i, g: a, b: n, a: o } = ee.parse(e), { r: s, g: l, b: c, a: h } = ee.parse(t), u = r / 100, f = u * 2 - 1, d = o - h, m = ((f * d === -1 ? f : (f + d) / (1 + f * d)) + 1) / 2, y = 1 - m, x = i * m + s * y, b = a * m + l * y, k = n * m + c * y, S = o * u + h * (1 - u);
  return Gr(x, b, k, S);
}, P = (e, t = 100) => {
  const r = ee.parse(e);
  return r.r = 255 - r.r, r.g = 255 - r.g, r.b = 255 - r.b, og(r, e, t);
};
var Nl = Object.defineProperty, p = (e, t) => Nl(e, "name", { value: t, configurable: !0 }), lg = (e, t) => {
  for (var r in t)
    Nl(e, r, { get: t[r], enumerable: !0 });
}, ne = {
  trace: 0,
  debug: 1,
  info: 2,
  warn: 3,
  error: 4,
  fatal: 5
}, F = {
  trace: /* @__PURE__ */ p((...e) => {
  }, "trace"),
  debug: /* @__PURE__ */ p((...e) => {
  }, "debug"),
  info: /* @__PURE__ */ p((...e) => {
  }, "info"),
  warn: /* @__PURE__ */ p((...e) => {
  }, "warn"),
  error: /* @__PURE__ */ p((...e) => {
  }, "error"),
  fatal: /* @__PURE__ */ p((...e) => {
  }, "fatal")
}, hs = /* @__PURE__ */ p(function(e = "fatal") {
  let t = ne.fatal;
  typeof e == "string" ? e.toLowerCase() in ne && (t = ne[e]) : typeof e == "number" && (t = e), F.trace = () => {
  }, F.debug = () => {
  }, F.info = () => {
  }, F.warn = () => {
  }, F.error = () => {
  }, F.fatal = () => {
  }, t <= ne.fatal && (F.fatal = console.error ? console.error.bind(console, Wt("FATAL"), "color: orange") : console.log.bind(console, "\x1B[35m", Wt("FATAL"))), t <= ne.error && (F.error = console.error ? console.error.bind(console, Wt("ERROR"), "color: orange") : console.log.bind(console, "\x1B[31m", Wt("ERROR"))), t <= ne.warn && (F.warn = console.warn ? console.warn.bind(console, Wt("WARN"), "color: orange") : console.log.bind(console, "\x1B[33m", Wt("WARN"))), t <= ne.info && (F.info = console.info ? console.info.bind(console, Wt("INFO"), "color: lightblue") : console.log.bind(console, "\x1B[34m", Wt("INFO"))), t <= ne.debug && (F.debug = console.debug ? console.debug.bind(console, Wt("DEBUG"), "color: lightgreen") : console.log.bind(console, "\x1B[32m", Wt("DEBUG"))), t <= ne.trace && (F.trace = console.debug ? console.debug.bind(console, Wt("TRACE"), "color: lightgreen") : console.log.bind(console, "\x1B[32m", Wt("TRACE")));
}, "setLogLevel"), Wt = /* @__PURE__ */ p((e) => `%c${tg().format("ss.SSS")} : ${e} : `, "format"), zl = /^-{3}\s*[\n\r](.*?)[\n\r]-{3}\s*[\n\r]+/s, Ur = /%{2}{\s*(?:(\w+)\s*:|(\w+))\s*(?:(\w+)|((?:(?!}%{2}).|\r?\n)*))?\s*(?:}%{2})?/gi, cg = /\s*%%.*\n/gm, sr, ql = (sr = class extends Error {
  constructor(t) {
    super(t), this.name = "UnknownDiagramError";
  }
}, p(sr, "UnknownDiagramError"), sr), ze = {}, us = /* @__PURE__ */ p(function(e, t) {
  e = e.replace(zl, "").replace(Ur, "").replace(cg, `
`);
  for (const [r, { detector: i }] of Object.entries(ze))
    if (i(e, t))
      return r;
  throw new ql(
    `No diagram type detected matching given configuration for text: ${e}`
  );
}, "detectType"), gn = /* @__PURE__ */ p((...e) => {
  for (const { id: t, detector: r, loader: i } of e)
    Wl(t, r, i);
}, "registerLazyLoadedDiagrams"), Wl = /* @__PURE__ */ p((e, t, r) => {
  ze[e] && F.warn(`Detector with key ${e} already exists. Overwriting.`), ze[e] = { detector: t, loader: r }, F.debug(`Detector with key ${e} added${r ? " with loader" : ""}`);
}, "addDetector"), hg = /* @__PURE__ */ p((e) => ze[e].loader, "getDiagramLoader"), mn = /* @__PURE__ */ p((e, t, { depth: r = 2, clobber: i = !1 } = {}) => {
  const a = { depth: r, clobber: i };
  return Array.isArray(t) && !Array.isArray(e) ? (t.forEach((n) => mn(e, n, a)), e) : Array.isArray(t) && Array.isArray(e) ? (t.forEach((n) => {
    e.includes(n) || e.push(n);
  }), e) : e === void 0 || r <= 0 ? e != null && typeof e == "object" && typeof t == "object" ? Object.assign(e, t) : t : (t !== void 0 && typeof e == "object" && typeof t == "object" && Object.keys(t).forEach((n) => {
    typeof t[n] == "object" && (e[n] === void 0 || typeof e[n] == "object") ? (e[n] === void 0 && (e[n] = Array.isArray(t[n]) ? [] : {}), e[n] = mn(e[n], t[n], { depth: r - 1, clobber: i })) : (i || typeof e[n] != "object" && typeof t[n] != "object") && (e[n] = t[n]);
  }), e);
}, "assignWithDepth"), vt = mn, Sa = "#ffffff", Ta = "#f2f2f2", $t = /* @__PURE__ */ p((e, t) => t ? v(e, { s: -40, l: 10 }) : v(e, { s: -40, l: -10 }), "mkBorder"), or, ug = (or = class {
  constructor() {
    this.background = "#f4f4f4", this.primaryColor = "#fff4dd", this.noteBkgColor = "#fff5ad", this.noteTextColor = "#333", this.THEME_COLOR_LIMIT = 12, this.fontFamily = '"trebuchet ms", verdana, arial, sans-serif', this.fontSize = "16px";
  }
  updateColors() {
    var r, i, a, n, o, s, l, c, h, u, f, d, g, m, y, x, b, k, S, w, C;
    if (this.primaryTextColor = this.primaryTextColor || (this.darkMode ? "#eee" : "#333"), this.secondaryColor = this.secondaryColor || v(this.primaryColor, { h: -120 }), this.tertiaryColor = this.tertiaryColor || v(this.primaryColor, { h: 180, l: 5 }), this.primaryBorderColor = this.primaryBorderColor || $t(this.primaryColor, this.darkMode), this.secondaryBorderColor = this.secondaryBorderColor || $t(this.secondaryColor, this.darkMode), this.tertiaryBorderColor = this.tertiaryBorderColor || $t(this.tertiaryColor, this.darkMode), this.noteBorderColor = this.noteBorderColor || $t(this.noteBkgColor, this.darkMode), this.noteBkgColor = this.noteBkgColor || "#fff5ad", this.noteTextColor = this.noteTextColor || "#333", this.secondaryTextColor = this.secondaryTextColor || P(this.secondaryColor), this.tertiaryTextColor = this.tertiaryTextColor || P(this.tertiaryColor), this.lineColor = this.lineColor || P(this.background), this.arrowheadColor = this.arrowheadColor || P(this.background), this.textColor = this.textColor || this.primaryTextColor, this.border2 = this.border2 || this.tertiaryBorderColor, this.nodeBkg = this.nodeBkg || this.primaryColor, this.mainBkg = this.mainBkg || this.primaryColor, this.nodeBorder = this.nodeBorder || this.primaryBorderColor, this.clusterBkg = this.clusterBkg || this.tertiaryColor, this.clusterBorder = this.clusterBorder || this.tertiaryBorderColor, this.defaultLinkColor = this.defaultLinkColor || this.lineColor, this.titleColor = this.titleColor || this.tertiaryTextColor, this.edgeLabelBackground = this.edgeLabelBackground || (this.darkMode ? X(this.secondaryColor, 30) : this.secondaryColor), this.nodeTextColor = this.nodeTextColor || this.primaryTextColor, this.actorBorder = this.actorBorder || this.primaryBorderColor, this.actorBkg = this.actorBkg || this.mainBkg, this.actorTextColor = this.actorTextColor || this.primaryTextColor, this.actorLineColor = this.actorLineColor || this.actorBorder, this.labelBoxBkgColor = this.labelBoxBkgColor || this.actorBkg, this.signalColor = this.signalColor || this.textColor, this.signalTextColor = this.signalTextColor || this.textColor, this.labelBoxBorderColor = this.labelBoxBorderColor || this.actorBorder, this.labelTextColor = this.labelTextColor || this.actorTextColor, this.loopTextColor = this.loopTextColor || this.actorTextColor, this.activationBorderColor = this.activationBorderColor || X(this.secondaryColor, 10), this.activationBkgColor = this.activationBkgColor || this.secondaryColor, this.sequenceNumberColor = this.sequenceNumberColor || P(this.lineColor), this.sectionBkgColor = this.sectionBkgColor || this.tertiaryColor, this.altSectionBkgColor = this.altSectionBkgColor || "white", this.sectionBkgColor = this.sectionBkgColor || this.secondaryColor, this.sectionBkgColor2 = this.sectionBkgColor2 || this.primaryColor, this.excludeBkgColor = this.excludeBkgColor || "#eeeeee", this.taskBorderColor = this.taskBorderColor || this.primaryBorderColor, this.taskBkgColor = this.taskBkgColor || this.primaryColor, this.activeTaskBorderColor = this.activeTaskBorderColor || this.primaryColor, this.activeTaskBkgColor = this.activeTaskBkgColor || z(this.primaryColor, 23), this.gridColor = this.gridColor || "lightgrey", this.doneTaskBkgColor = this.doneTaskBkgColor || "lightgrey", this.doneTaskBorderColor = this.doneTaskBorderColor || "grey", this.critBorderColor = this.critBorderColor || "#ff8888", this.critBkgColor = this.critBkgColor || "red", this.todayLineColor = this.todayLineColor || "red", this.vertLineColor = this.vertLineColor || "navy", this.taskTextColor = this.taskTextColor || this.textColor, this.taskTextOutsideColor = this.taskTextOutsideColor || this.textColor, this.taskTextLightColor = this.taskTextLightColor || this.textColor, this.taskTextColor = this.taskTextColor || this.primaryTextColor, this.taskTextDarkColor = this.taskTextDarkColor || this.textColor, this.taskTextClickableColor = this.taskTextClickableColor || "#003163", this.personBorder = this.personBorder || this.primaryBorderColor, this.personBkg = this.personBkg || this.mainBkg, this.darkMode ? (this.rowOdd = this.rowOdd || X(this.mainBkg, 5) || "#ffffff", this.rowEven = this.rowEven || X(this.mainBkg, 10)) : (this.rowOdd = this.rowOdd || z(this.mainBkg, 75) || "#ffffff", this.rowEven = this.rowEven || z(this.mainBkg, 5)), this.transitionColor = this.transitionColor || this.lineColor, this.transitionLabelColor = this.transitionLabelColor || this.textColor, this.stateLabelColor = this.stateLabelColor || this.stateBkg || this.primaryTextColor, this.stateBkg = this.stateBkg || this.mainBkg, this.labelBackgroundColor = this.labelBackgroundColor || this.stateBkg, this.compositeBackground = this.compositeBackground || this.background || this.tertiaryColor, this.altBackground = this.altBackground || this.tertiaryColor, this.compositeTitleBackground = this.compositeTitleBackground || this.mainBkg, this.compositeBorder = this.compositeBorder || this.nodeBorder, this.innerEndBackground = this.nodeBorder, this.errorBkgColor = this.errorBkgColor || this.tertiaryColor, this.errorTextColor = this.errorTextColor || this.tertiaryTextColor, this.transitionColor = this.transitionColor || this.lineColor, this.specialStateColor = this.lineColor, this.cScale0 = this.cScale0 || this.primaryColor, this.cScale1 = this.cScale1 || this.secondaryColor, this.cScale2 = this.cScale2 || this.tertiaryColor, this.cScale3 = this.cScale3 || v(this.primaryColor, { h: 30 }), this.cScale4 = this.cScale4 || v(this.primaryColor, { h: 60 }), this.cScale5 = this.cScale5 || v(this.primaryColor, { h: 90 }), this.cScale6 = this.cScale6 || v(this.primaryColor, { h: 120 }), this.cScale7 = this.cScale7 || v(this.primaryColor, { h: 150 }), this.cScale8 = this.cScale8 || v(this.primaryColor, { h: 210, l: 150 }), this.cScale9 = this.cScale9 || v(this.primaryColor, { h: 270 }), this.cScale10 = this.cScale10 || v(this.primaryColor, { h: 300 }), this.cScale11 = this.cScale11 || v(this.primaryColor, { h: 330 }), this.darkMode)
      for (let _ = 0; _ < this.THEME_COLOR_LIMIT; _++)
        this["cScale" + _] = X(this["cScale" + _], 75);
    else
      for (let _ = 0; _ < this.THEME_COLOR_LIMIT; _++)
        this["cScale" + _] = X(this["cScale" + _], 25);
    for (let _ = 0; _ < this.THEME_COLOR_LIMIT; _++)
      this["cScaleInv" + _] = this["cScaleInv" + _] || P(this["cScale" + _]);
    for (let _ = 0; _ < this.THEME_COLOR_LIMIT; _++)
      this.darkMode ? this["cScalePeer" + _] = this["cScalePeer" + _] || z(this["cScale" + _], 10) : this["cScalePeer" + _] = this["cScalePeer" + _] || X(this["cScale" + _], 10);
    this.scaleLabelColor = this.scaleLabelColor || this.labelTextColor;
    for (let _ = 0; _ < this.THEME_COLOR_LIMIT; _++)
      this["cScaleLabel" + _] = this["cScaleLabel" + _] || this.scaleLabelColor;
    const t = this.darkMode ? -4 : -1;
    for (let _ = 0; _ < 5; _++)
      this["surface" + _] = this["surface" + _] || v(this.mainBkg, { h: 180, s: -15, l: t * (5 + _ * 3) }), this["surfacePeer" + _] = this["surfacePeer" + _] || v(this.mainBkg, { h: 180, s: -15, l: t * (8 + _ * 3) });
    this.classText = this.classText || this.textColor, this.fillType0 = this.fillType0 || this.primaryColor, this.fillType1 = this.fillType1 || this.secondaryColor, this.fillType2 = this.fillType2 || v(this.primaryColor, { h: 64 }), this.fillType3 = this.fillType3 || v(this.secondaryColor, { h: 64 }), this.fillType4 = this.fillType4 || v(this.primaryColor, { h: -64 }), this.fillType5 = this.fillType5 || v(this.secondaryColor, { h: -64 }), this.fillType6 = this.fillType6 || v(this.primaryColor, { h: 128 }), this.fillType7 = this.fillType7 || v(this.secondaryColor, { h: 128 }), this.pie1 = this.pie1 || this.primaryColor, this.pie2 = this.pie2 || this.secondaryColor, this.pie3 = this.pie3 || this.tertiaryColor, this.pie4 = this.pie4 || v(this.primaryColor, { l: -10 }), this.pie5 = this.pie5 || v(this.secondaryColor, { l: -10 }), this.pie6 = this.pie6 || v(this.tertiaryColor, { l: -10 }), this.pie7 = this.pie7 || v(this.primaryColor, { h: 60, l: -10 }), this.pie8 = this.pie8 || v(this.primaryColor, { h: -60, l: -10 }), this.pie9 = this.pie9 || v(this.primaryColor, { h: 120, l: 0 }), this.pie10 = this.pie10 || v(this.primaryColor, { h: 60, l: -20 }), this.pie11 = this.pie11 || v(this.primaryColor, { h: -60, l: -20 }), this.pie12 = this.pie12 || v(this.primaryColor, { h: 120, l: -10 }), this.pieTitleTextSize = this.pieTitleTextSize || "25px", this.pieTitleTextColor = this.pieTitleTextColor || this.taskTextDarkColor, this.pieSectionTextSize = this.pieSectionTextSize || "17px", this.pieSectionTextColor = this.pieSectionTextColor || this.textColor, this.pieLegendTextSize = this.pieLegendTextSize || "17px", this.pieLegendTextColor = this.pieLegendTextColor || this.taskTextDarkColor, this.pieStrokeColor = this.pieStrokeColor || "black", this.pieStrokeWidth = this.pieStrokeWidth || "2px", this.pieOuterStrokeWidth = this.pieOuterStrokeWidth || "2px", this.pieOuterStrokeColor = this.pieOuterStrokeColor || "black", this.pieOpacity = this.pieOpacity || "0.7", this.radar = {
      axisColor: ((r = this.radar) == null ? void 0 : r.axisColor) || this.lineColor,
      axisStrokeWidth: ((i = this.radar) == null ? void 0 : i.axisStrokeWidth) || 2,
      axisLabelFontSize: ((a = this.radar) == null ? void 0 : a.axisLabelFontSize) || 12,
      curveOpacity: ((n = this.radar) == null ? void 0 : n.curveOpacity) || 0.5,
      curveStrokeWidth: ((o = this.radar) == null ? void 0 : o.curveStrokeWidth) || 2,
      graticuleColor: ((s = this.radar) == null ? void 0 : s.graticuleColor) || "#DEDEDE",
      graticuleStrokeWidth: ((l = this.radar) == null ? void 0 : l.graticuleStrokeWidth) || 1,
      graticuleOpacity: ((c = this.radar) == null ? void 0 : c.graticuleOpacity) || 0.3,
      legendBoxSize: ((h = this.radar) == null ? void 0 : h.legendBoxSize) || 12,
      legendFontSize: ((u = this.radar) == null ? void 0 : u.legendFontSize) || 12
    }, this.archEdgeColor = this.archEdgeColor || "#777", this.archEdgeArrowColor = this.archEdgeArrowColor || "#777", this.archEdgeWidth = this.archEdgeWidth || "3", this.archGroupBorderColor = this.archGroupBorderColor || "#000", this.archGroupBorderWidth = this.archGroupBorderWidth || "2px", this.quadrant1Fill = this.quadrant1Fill || this.primaryColor, this.quadrant2Fill = this.quadrant2Fill || v(this.primaryColor, { r: 5, g: 5, b: 5 }), this.quadrant3Fill = this.quadrant3Fill || v(this.primaryColor, { r: 10, g: 10, b: 10 }), this.quadrant4Fill = this.quadrant4Fill || v(this.primaryColor, { r: 15, g: 15, b: 15 }), this.quadrant1TextFill = this.quadrant1TextFill || this.primaryTextColor, this.quadrant2TextFill = this.quadrant2TextFill || v(this.primaryTextColor, { r: -5, g: -5, b: -5 }), this.quadrant3TextFill = this.quadrant3TextFill || v(this.primaryTextColor, { r: -10, g: -10, b: -10 }), this.quadrant4TextFill = this.quadrant4TextFill || v(this.primaryTextColor, { r: -15, g: -15, b: -15 }), this.quadrantPointFill = this.quadrantPointFill || ci(this.quadrant1Fill) ? z(this.quadrant1Fill) : X(this.quadrant1Fill), this.quadrantPointTextFill = this.quadrantPointTextFill || this.primaryTextColor, this.quadrantXAxisTextFill = this.quadrantXAxisTextFill || this.primaryTextColor, this.quadrantYAxisTextFill = this.quadrantYAxisTextFill || this.primaryTextColor, this.quadrantInternalBorderStrokeFill = this.quadrantInternalBorderStrokeFill || this.primaryBorderColor, this.quadrantExternalBorderStrokeFill = this.quadrantExternalBorderStrokeFill || this.primaryBorderColor, this.quadrantTitleFill = this.quadrantTitleFill || this.primaryTextColor, this.xyChart = {
      backgroundColor: ((f = this.xyChart) == null ? void 0 : f.backgroundColor) || this.background,
      titleColor: ((d = this.xyChart) == null ? void 0 : d.titleColor) || this.primaryTextColor,
      xAxisTitleColor: ((g = this.xyChart) == null ? void 0 : g.xAxisTitleColor) || this.primaryTextColor,
      xAxisLabelColor: ((m = this.xyChart) == null ? void 0 : m.xAxisLabelColor) || this.primaryTextColor,
      xAxisTickColor: ((y = this.xyChart) == null ? void 0 : y.xAxisTickColor) || this.primaryTextColor,
      xAxisLineColor: ((x = this.xyChart) == null ? void 0 : x.xAxisLineColor) || this.primaryTextColor,
      yAxisTitleColor: ((b = this.xyChart) == null ? void 0 : b.yAxisTitleColor) || this.primaryTextColor,
      yAxisLabelColor: ((k = this.xyChart) == null ? void 0 : k.yAxisLabelColor) || this.primaryTextColor,
      yAxisTickColor: ((S = this.xyChart) == null ? void 0 : S.yAxisTickColor) || this.primaryTextColor,
      yAxisLineColor: ((w = this.xyChart) == null ? void 0 : w.yAxisLineColor) || this.primaryTextColor,
      plotColorPalette: ((C = this.xyChart) == null ? void 0 : C.plotColorPalette) || "#FFF4DD,#FFD8B1,#FFA07A,#ECEFF1,#D6DBDF,#C3E0A8,#FFB6A4,#FFD74D,#738FA7,#FFFFF0"
    }, this.requirementBackground = this.requirementBackground || this.primaryColor, this.requirementBorderColor = this.requirementBorderColor || this.primaryBorderColor, this.requirementBorderSize = this.requirementBorderSize || "1", this.requirementTextColor = this.requirementTextColor || this.primaryTextColor, this.relationColor = this.relationColor || this.lineColor, this.relationLabelBackground = this.relationLabelBackground || (this.darkMode ? X(this.secondaryColor, 30) : this.secondaryColor), this.relationLabelColor = this.relationLabelColor || this.actorTextColor, this.git0 = this.git0 || this.primaryColor, this.git1 = this.git1 || this.secondaryColor, this.git2 = this.git2 || this.tertiaryColor, this.git3 = this.git3 || v(this.primaryColor, { h: -30 }), this.git4 = this.git4 || v(this.primaryColor, { h: -60 }), this.git5 = this.git5 || v(this.primaryColor, { h: -90 }), this.git6 = this.git6 || v(this.primaryColor, { h: 60 }), this.git7 = this.git7 || v(this.primaryColor, { h: 120 }), this.darkMode ? (this.git0 = z(this.git0, 25), this.git1 = z(this.git1, 25), this.git2 = z(this.git2, 25), this.git3 = z(this.git3, 25), this.git4 = z(this.git4, 25), this.git5 = z(this.git5, 25), this.git6 = z(this.git6, 25), this.git7 = z(this.git7, 25)) : (this.git0 = X(this.git0, 25), this.git1 = X(this.git1, 25), this.git2 = X(this.git2, 25), this.git3 = X(this.git3, 25), this.git4 = X(this.git4, 25), this.git5 = X(this.git5, 25), this.git6 = X(this.git6, 25), this.git7 = X(this.git7, 25)), this.gitInv0 = this.gitInv0 || P(this.git0), this.gitInv1 = this.gitInv1 || P(this.git1), this.gitInv2 = this.gitInv2 || P(this.git2), this.gitInv3 = this.gitInv3 || P(this.git3), this.gitInv4 = this.gitInv4 || P(this.git4), this.gitInv5 = this.gitInv5 || P(this.git5), this.gitInv6 = this.gitInv6 || P(this.git6), this.gitInv7 = this.gitInv7 || P(this.git7), this.branchLabelColor = this.branchLabelColor || (this.darkMode ? "black" : this.labelTextColor), this.gitBranchLabel0 = this.gitBranchLabel0 || this.branchLabelColor, this.gitBranchLabel1 = this.gitBranchLabel1 || this.branchLabelColor, this.gitBranchLabel2 = this.gitBranchLabel2 || this.branchLabelColor, this.gitBranchLabel3 = this.gitBranchLabel3 || this.branchLabelColor, this.gitBranchLabel4 = this.gitBranchLabel4 || this.branchLabelColor, this.gitBranchLabel5 = this.gitBranchLabel5 || this.branchLabelColor, this.gitBranchLabel6 = this.gitBranchLabel6 || this.branchLabelColor, this.gitBranchLabel7 = this.gitBranchLabel7 || this.branchLabelColor, this.tagLabelColor = this.tagLabelColor || this.primaryTextColor, this.tagLabelBackground = this.tagLabelBackground || this.primaryColor, this.tagLabelBorder = this.tagBorder || this.primaryBorderColor, this.tagLabelFontSize = this.tagLabelFontSize || "10px", this.commitLabelColor = this.commitLabelColor || this.secondaryTextColor, this.commitLabelBackground = this.commitLabelBackground || this.secondaryColor, this.commitLabelFontSize = this.commitLabelFontSize || "10px", this.attributeBackgroundColorOdd = this.attributeBackgroundColorOdd || Sa, this.attributeBackgroundColorEven = this.attributeBackgroundColorEven || Ta;
  }
  calculate(t) {
    if (typeof t != "object") {
      this.updateColors();
      return;
    }
    const r = Object.keys(t);
    r.forEach((i) => {
      this[i] = t[i];
    }), this.updateColors(), r.forEach((i) => {
      this[i] = t[i];
    });
  }
}, p(or, "Theme"), or), fg = /* @__PURE__ */ p((e) => {
  const t = new ug();
  return t.calculate(e), t;
}, "getThemeVariables"), lr, dg = (lr = class {
  constructor() {
    this.background = "#333", this.primaryColor = "#1f2020", this.secondaryColor = z(this.primaryColor, 16), this.tertiaryColor = v(this.primaryColor, { h: -160 }), this.primaryBorderColor = P(this.background), this.secondaryBorderColor = $t(this.secondaryColor, this.darkMode), this.tertiaryBorderColor = $t(this.tertiaryColor, this.darkMode), this.primaryTextColor = P(this.primaryColor), this.secondaryTextColor = P(this.secondaryColor), this.tertiaryTextColor = P(this.tertiaryColor), this.lineColor = P(this.background), this.textColor = P(this.background), this.mainBkg = "#1f2020", this.secondBkg = "calculated", this.mainContrastColor = "lightgrey", this.darkTextColor = z(P("#323D47"), 10), this.lineColor = "calculated", this.border1 = "#ccc", this.border2 = Gr(255, 255, 255, 0.25), this.arrowheadColor = "calculated", this.fontFamily = '"trebuchet ms", verdana, arial, sans-serif', this.fontSize = "16px", this.labelBackground = "#181818", this.textColor = "#ccc", this.THEME_COLOR_LIMIT = 12, this.nodeBkg = "calculated", this.nodeBorder = "calculated", this.clusterBkg = "calculated", this.clusterBorder = "calculated", this.defaultLinkColor = "calculated", this.titleColor = "#F9FFFE", this.edgeLabelBackground = "calculated", this.actorBorder = "calculated", this.actorBkg = "calculated", this.actorTextColor = "calculated", this.actorLineColor = "calculated", this.signalColor = "calculated", this.signalTextColor = "calculated", this.labelBoxBkgColor = "calculated", this.labelBoxBorderColor = "calculated", this.labelTextColor = "calculated", this.loopTextColor = "calculated", this.noteBorderColor = "calculated", this.noteBkgColor = "#fff5ad", this.noteTextColor = "calculated", this.activationBorderColor = "calculated", this.activationBkgColor = "calculated", this.sequenceNumberColor = "black", this.sectionBkgColor = X("#EAE8D9", 30), this.altSectionBkgColor = "calculated", this.sectionBkgColor2 = "#EAE8D9", this.excludeBkgColor = X(this.sectionBkgColor, 10), this.taskBorderColor = Gr(255, 255, 255, 70), this.taskBkgColor = "calculated", this.taskTextColor = "calculated", this.taskTextLightColor = "calculated", this.taskTextOutsideColor = "calculated", this.taskTextClickableColor = "#003163", this.activeTaskBorderColor = Gr(255, 255, 255, 50), this.activeTaskBkgColor = "#81B1DB", this.gridColor = "calculated", this.doneTaskBkgColor = "calculated", this.doneTaskBorderColor = "grey", this.critBorderColor = "#E83737", this.critBkgColor = "#E83737", this.taskTextDarkColor = "calculated", this.todayLineColor = "#DB5757", this.vertLineColor = "#00BFFF", this.personBorder = this.primaryBorderColor, this.personBkg = this.mainBkg, this.archEdgeColor = "calculated", this.archEdgeArrowColor = "calculated", this.archEdgeWidth = "3", this.archGroupBorderColor = this.primaryBorderColor, this.archGroupBorderWidth = "2px", this.rowOdd = this.rowOdd || z(this.mainBkg, 5) || "#ffffff", this.rowEven = this.rowEven || X(this.mainBkg, 10), this.labelColor = "calculated", this.errorBkgColor = "#a44141", this.errorTextColor = "#ddd";
  }
  updateColors() {
    var t, r, i, a, n, o, s, l, c, h, u, f, d, g, m, y, x, b, k, S, w;
    this.secondBkg = z(this.mainBkg, 16), this.lineColor = this.mainContrastColor, this.arrowheadColor = this.mainContrastColor, this.nodeBkg = this.mainBkg, this.nodeBorder = this.border1, this.clusterBkg = this.secondBkg, this.clusterBorder = this.border2, this.defaultLinkColor = this.lineColor, this.edgeLabelBackground = z(this.labelBackground, 25), this.actorBorder = this.border1, this.actorBkg = this.mainBkg, this.actorTextColor = this.mainContrastColor, this.actorLineColor = this.actorBorder, this.signalColor = this.mainContrastColor, this.signalTextColor = this.mainContrastColor, this.labelBoxBkgColor = this.actorBkg, this.labelBoxBorderColor = this.actorBorder, this.labelTextColor = this.mainContrastColor, this.loopTextColor = this.mainContrastColor, this.noteBorderColor = this.secondaryBorderColor, this.noteBkgColor = this.secondBkg, this.noteTextColor = this.secondaryTextColor, this.activationBorderColor = this.border1, this.activationBkgColor = this.secondBkg, this.altSectionBkgColor = this.background, this.taskBkgColor = z(this.mainBkg, 23), this.taskTextColor = this.darkTextColor, this.taskTextLightColor = this.mainContrastColor, this.taskTextOutsideColor = this.taskTextLightColor, this.gridColor = this.mainContrastColor, this.doneTaskBkgColor = this.mainContrastColor, this.taskTextDarkColor = this.darkTextColor, this.archEdgeColor = this.lineColor, this.archEdgeArrowColor = this.lineColor, this.transitionColor = this.transitionColor || this.lineColor, this.transitionLabelColor = this.transitionLabelColor || this.textColor, this.stateLabelColor = this.stateLabelColor || this.stateBkg || this.primaryTextColor, this.stateBkg = this.stateBkg || this.mainBkg, this.labelBackgroundColor = this.labelBackgroundColor || this.stateBkg, this.compositeBackground = this.compositeBackground || this.background || this.tertiaryColor, this.altBackground = this.altBackground || "#555", this.compositeTitleBackground = this.compositeTitleBackground || this.mainBkg, this.compositeBorder = this.compositeBorder || this.nodeBorder, this.innerEndBackground = this.primaryBorderColor, this.specialStateColor = "#f4f4f4", this.errorBkgColor = this.errorBkgColor || this.tertiaryColor, this.errorTextColor = this.errorTextColor || this.tertiaryTextColor, this.fillType0 = this.primaryColor, this.fillType1 = this.secondaryColor, this.fillType2 = v(this.primaryColor, { h: 64 }), this.fillType3 = v(this.secondaryColor, { h: 64 }), this.fillType4 = v(this.primaryColor, { h: -64 }), this.fillType5 = v(this.secondaryColor, { h: -64 }), this.fillType6 = v(this.primaryColor, { h: 128 }), this.fillType7 = v(this.secondaryColor, { h: 128 }), this.cScale1 = this.cScale1 || "#0b0000", this.cScale2 = this.cScale2 || "#4d1037", this.cScale3 = this.cScale3 || "#3f5258", this.cScale4 = this.cScale4 || "#4f2f1b", this.cScale5 = this.cScale5 || "#6e0a0a", this.cScale6 = this.cScale6 || "#3b0048", this.cScale7 = this.cScale7 || "#995a01", this.cScale8 = this.cScale8 || "#154706", this.cScale9 = this.cScale9 || "#161722", this.cScale10 = this.cScale10 || "#00296f", this.cScale11 = this.cScale11 || "#01629c", this.cScale12 = this.cScale12 || "#010029", this.cScale0 = this.cScale0 || this.primaryColor, this.cScale1 = this.cScale1 || this.secondaryColor, this.cScale2 = this.cScale2 || this.tertiaryColor, this.cScale3 = this.cScale3 || v(this.primaryColor, { h: 30 }), this.cScale4 = this.cScale4 || v(this.primaryColor, { h: 60 }), this.cScale5 = this.cScale5 || v(this.primaryColor, { h: 90 }), this.cScale6 = this.cScale6 || v(this.primaryColor, { h: 120 }), this.cScale7 = this.cScale7 || v(this.primaryColor, { h: 150 }), this.cScale8 = this.cScale8 || v(this.primaryColor, { h: 210 }), this.cScale9 = this.cScale9 || v(this.primaryColor, { h: 270 }), this.cScale10 = this.cScale10 || v(this.primaryColor, { h: 300 }), this.cScale11 = this.cScale11 || v(this.primaryColor, { h: 330 });
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScaleInv" + C] = this["cScaleInv" + C] || P(this["cScale" + C]);
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScalePeer" + C] = this["cScalePeer" + C] || z(this["cScale" + C], 10);
    for (let C = 0; C < 5; C++)
      this["surface" + C] = this["surface" + C] || v(this.mainBkg, { h: 30, s: -30, l: -(-10 + C * 4) }), this["surfacePeer" + C] = this["surfacePeer" + C] || v(this.mainBkg, { h: 30, s: -30, l: -(-7 + C * 4) });
    this.scaleLabelColor = this.scaleLabelColor || (this.darkMode ? "black" : this.labelTextColor);
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScaleLabel" + C] = this["cScaleLabel" + C] || this.scaleLabelColor;
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["pie" + C] = this["cScale" + C];
    this.pieTitleTextSize = this.pieTitleTextSize || "25px", this.pieTitleTextColor = this.pieTitleTextColor || this.taskTextDarkColor, this.pieSectionTextSize = this.pieSectionTextSize || "17px", this.pieSectionTextColor = this.pieSectionTextColor || this.textColor, this.pieLegendTextSize = this.pieLegendTextSize || "17px", this.pieLegendTextColor = this.pieLegendTextColor || this.taskTextDarkColor, this.pieStrokeColor = this.pieStrokeColor || "black", this.pieStrokeWidth = this.pieStrokeWidth || "2px", this.pieOuterStrokeWidth = this.pieOuterStrokeWidth || "2px", this.pieOuterStrokeColor = this.pieOuterStrokeColor || "black", this.pieOpacity = this.pieOpacity || "0.7", this.quadrant1Fill = this.quadrant1Fill || this.primaryColor, this.quadrant2Fill = this.quadrant2Fill || v(this.primaryColor, { r: 5, g: 5, b: 5 }), this.quadrant3Fill = this.quadrant3Fill || v(this.primaryColor, { r: 10, g: 10, b: 10 }), this.quadrant4Fill = this.quadrant4Fill || v(this.primaryColor, { r: 15, g: 15, b: 15 }), this.quadrant1TextFill = this.quadrant1TextFill || this.primaryTextColor, this.quadrant2TextFill = this.quadrant2TextFill || v(this.primaryTextColor, { r: -5, g: -5, b: -5 }), this.quadrant3TextFill = this.quadrant3TextFill || v(this.primaryTextColor, { r: -10, g: -10, b: -10 }), this.quadrant4TextFill = this.quadrant4TextFill || v(this.primaryTextColor, { r: -15, g: -15, b: -15 }), this.quadrantPointFill = this.quadrantPointFill || ci(this.quadrant1Fill) ? z(this.quadrant1Fill) : X(this.quadrant1Fill), this.quadrantPointTextFill = this.quadrantPointTextFill || this.primaryTextColor, this.quadrantXAxisTextFill = this.quadrantXAxisTextFill || this.primaryTextColor, this.quadrantYAxisTextFill = this.quadrantYAxisTextFill || this.primaryTextColor, this.quadrantInternalBorderStrokeFill = this.quadrantInternalBorderStrokeFill || this.primaryBorderColor, this.quadrantExternalBorderStrokeFill = this.quadrantExternalBorderStrokeFill || this.primaryBorderColor, this.quadrantTitleFill = this.quadrantTitleFill || this.primaryTextColor, this.xyChart = {
      backgroundColor: ((t = this.xyChart) == null ? void 0 : t.backgroundColor) || this.background,
      titleColor: ((r = this.xyChart) == null ? void 0 : r.titleColor) || this.primaryTextColor,
      xAxisTitleColor: ((i = this.xyChart) == null ? void 0 : i.xAxisTitleColor) || this.primaryTextColor,
      xAxisLabelColor: ((a = this.xyChart) == null ? void 0 : a.xAxisLabelColor) || this.primaryTextColor,
      xAxisTickColor: ((n = this.xyChart) == null ? void 0 : n.xAxisTickColor) || this.primaryTextColor,
      xAxisLineColor: ((o = this.xyChart) == null ? void 0 : o.xAxisLineColor) || this.primaryTextColor,
      yAxisTitleColor: ((s = this.xyChart) == null ? void 0 : s.yAxisTitleColor) || this.primaryTextColor,
      yAxisLabelColor: ((l = this.xyChart) == null ? void 0 : l.yAxisLabelColor) || this.primaryTextColor,
      yAxisTickColor: ((c = this.xyChart) == null ? void 0 : c.yAxisTickColor) || this.primaryTextColor,
      yAxisLineColor: ((h = this.xyChart) == null ? void 0 : h.yAxisLineColor) || this.primaryTextColor,
      plotColorPalette: ((u = this.xyChart) == null ? void 0 : u.plotColorPalette) || "#3498db,#2ecc71,#e74c3c,#f1c40f,#bdc3c7,#ffffff,#34495e,#9b59b6,#1abc9c,#e67e22"
    }, this.packet = {
      startByteColor: this.primaryTextColor,
      endByteColor: this.primaryTextColor,
      labelColor: this.primaryTextColor,
      titleColor: this.primaryTextColor,
      blockStrokeColor: this.primaryTextColor,
      blockFillColor: this.background
    }, this.radar = {
      axisColor: ((f = this.radar) == null ? void 0 : f.axisColor) || this.lineColor,
      axisStrokeWidth: ((d = this.radar) == null ? void 0 : d.axisStrokeWidth) || 2,
      axisLabelFontSize: ((g = this.radar) == null ? void 0 : g.axisLabelFontSize) || 12,
      curveOpacity: ((m = this.radar) == null ? void 0 : m.curveOpacity) || 0.5,
      curveStrokeWidth: ((y = this.radar) == null ? void 0 : y.curveStrokeWidth) || 2,
      graticuleColor: ((x = this.radar) == null ? void 0 : x.graticuleColor) || "#DEDEDE",
      graticuleStrokeWidth: ((b = this.radar) == null ? void 0 : b.graticuleStrokeWidth) || 1,
      graticuleOpacity: ((k = this.radar) == null ? void 0 : k.graticuleOpacity) || 0.3,
      legendBoxSize: ((S = this.radar) == null ? void 0 : S.legendBoxSize) || 12,
      legendFontSize: ((w = this.radar) == null ? void 0 : w.legendFontSize) || 12
    }, this.classText = this.primaryTextColor, this.requirementBackground = this.requirementBackground || this.primaryColor, this.requirementBorderColor = this.requirementBorderColor || this.primaryBorderColor, this.requirementBorderSize = this.requirementBorderSize || "1", this.requirementTextColor = this.requirementTextColor || this.primaryTextColor, this.relationColor = this.relationColor || this.lineColor, this.relationLabelBackground = this.relationLabelBackground || (this.darkMode ? X(this.secondaryColor, 30) : this.secondaryColor), this.relationLabelColor = this.relationLabelColor || this.actorTextColor, this.git0 = z(this.secondaryColor, 20), this.git1 = z(this.pie2 || this.secondaryColor, 20), this.git2 = z(this.pie3 || this.tertiaryColor, 20), this.git3 = z(this.pie4 || v(this.primaryColor, { h: -30 }), 20), this.git4 = z(this.pie5 || v(this.primaryColor, { h: -60 }), 20), this.git5 = z(this.pie6 || v(this.primaryColor, { h: -90 }), 10), this.git6 = z(this.pie7 || v(this.primaryColor, { h: 60 }), 10), this.git7 = z(this.pie8 || v(this.primaryColor, { h: 120 }), 20), this.gitInv0 = this.gitInv0 || P(this.git0), this.gitInv1 = this.gitInv1 || P(this.git1), this.gitInv2 = this.gitInv2 || P(this.git2), this.gitInv3 = this.gitInv3 || P(this.git3), this.gitInv4 = this.gitInv4 || P(this.git4), this.gitInv5 = this.gitInv5 || P(this.git5), this.gitInv6 = this.gitInv6 || P(this.git6), this.gitInv7 = this.gitInv7 || P(this.git7), this.gitBranchLabel0 = this.gitBranchLabel0 || P(this.labelTextColor), this.gitBranchLabel1 = this.gitBranchLabel1 || this.labelTextColor, this.gitBranchLabel2 = this.gitBranchLabel2 || this.labelTextColor, this.gitBranchLabel3 = this.gitBranchLabel3 || P(this.labelTextColor), this.gitBranchLabel4 = this.gitBranchLabel4 || this.labelTextColor, this.gitBranchLabel5 = this.gitBranchLabel5 || this.labelTextColor, this.gitBranchLabel6 = this.gitBranchLabel6 || this.labelTextColor, this.gitBranchLabel7 = this.gitBranchLabel7 || this.labelTextColor, this.tagLabelColor = this.tagLabelColor || this.primaryTextColor, this.tagLabelBackground = this.tagLabelBackground || this.primaryColor, this.tagLabelBorder = this.tagBorder || this.primaryBorderColor, this.tagLabelFontSize = this.tagLabelFontSize || "10px", this.commitLabelColor = this.commitLabelColor || this.secondaryTextColor, this.commitLabelBackground = this.commitLabelBackground || this.secondaryColor, this.commitLabelFontSize = this.commitLabelFontSize || "10px", this.attributeBackgroundColorOdd = this.attributeBackgroundColorOdd || z(this.background, 12), this.attributeBackgroundColorEven = this.attributeBackgroundColorEven || z(this.background, 2), this.nodeBorder = this.nodeBorder || "#999";
  }
  calculate(t) {
    if (typeof t != "object") {
      this.updateColors();
      return;
    }
    const r = Object.keys(t);
    r.forEach((i) => {
      this[i] = t[i];
    }), this.updateColors(), r.forEach((i) => {
      this[i] = t[i];
    });
  }
}, p(lr, "Theme"), lr), pg = /* @__PURE__ */ p((e) => {
  const t = new dg();
  return t.calculate(e), t;
}, "getThemeVariables"), cr, gg = (cr = class {
  constructor() {
    this.background = "#f4f4f4", this.primaryColor = "#ECECFF", this.secondaryColor = v(this.primaryColor, { h: 120 }), this.secondaryColor = "#ffffde", this.tertiaryColor = v(this.primaryColor, { h: -160 }), this.primaryBorderColor = $t(this.primaryColor, this.darkMode), this.secondaryBorderColor = $t(this.secondaryColor, this.darkMode), this.tertiaryBorderColor = $t(this.tertiaryColor, this.darkMode), this.primaryTextColor = P(this.primaryColor), this.secondaryTextColor = P(this.secondaryColor), this.tertiaryTextColor = P(this.tertiaryColor), this.lineColor = P(this.background), this.textColor = P(this.background), this.background = "white", this.mainBkg = "#ECECFF", this.secondBkg = "#ffffde", this.lineColor = "#333333", this.border1 = "#9370DB", this.border2 = "#aaaa33", this.arrowheadColor = "#333333", this.fontFamily = '"trebuchet ms", verdana, arial, sans-serif', this.fontSize = "16px", this.labelBackground = "rgba(232,232,232, 0.8)", this.textColor = "#333", this.THEME_COLOR_LIMIT = 12, this.nodeBkg = "calculated", this.nodeBorder = "calculated", this.clusterBkg = "calculated", this.clusterBorder = "calculated", this.defaultLinkColor = "calculated", this.titleColor = "calculated", this.edgeLabelBackground = "calculated", this.actorBorder = "calculated", this.actorBkg = "calculated", this.actorTextColor = "black", this.actorLineColor = "calculated", this.signalColor = "calculated", this.signalTextColor = "calculated", this.labelBoxBkgColor = "calculated", this.labelBoxBorderColor = "calculated", this.labelTextColor = "calculated", this.loopTextColor = "calculated", this.noteBorderColor = "calculated", this.noteBkgColor = "#fff5ad", this.noteTextColor = "calculated", this.activationBorderColor = "#666", this.activationBkgColor = "#f4f4f4", this.sequenceNumberColor = "white", this.sectionBkgColor = "calculated", this.altSectionBkgColor = "calculated", this.sectionBkgColor2 = "calculated", this.excludeBkgColor = "#eeeeee", this.taskBorderColor = "calculated", this.taskBkgColor = "calculated", this.taskTextLightColor = "calculated", this.taskTextColor = this.taskTextLightColor, this.taskTextDarkColor = "calculated", this.taskTextOutsideColor = this.taskTextDarkColor, this.taskTextClickableColor = "calculated", this.activeTaskBorderColor = "calculated", this.activeTaskBkgColor = "calculated", this.gridColor = "calculated", this.doneTaskBkgColor = "calculated", this.doneTaskBorderColor = "calculated", this.critBorderColor = "calculated", this.critBkgColor = "calculated", this.todayLineColor = "calculated", this.vertLineColor = "calculated", this.sectionBkgColor = Gr(102, 102, 255, 0.49), this.altSectionBkgColor = "white", this.sectionBkgColor2 = "#fff400", this.taskBorderColor = "#534fbc", this.taskBkgColor = "#8a90dd", this.taskTextLightColor = "white", this.taskTextColor = "calculated", this.taskTextDarkColor = "black", this.taskTextOutsideColor = "calculated", this.taskTextClickableColor = "#003163", this.activeTaskBorderColor = "#534fbc", this.activeTaskBkgColor = "#bfc7ff", this.gridColor = "lightgrey", this.doneTaskBkgColor = "lightgrey", this.doneTaskBorderColor = "grey", this.critBorderColor = "#ff8888", this.critBkgColor = "red", this.todayLineColor = "red", this.vertLineColor = "navy", this.personBorder = this.primaryBorderColor, this.personBkg = this.mainBkg, this.archEdgeColor = "calculated", this.archEdgeArrowColor = "calculated", this.archEdgeWidth = "3", this.archGroupBorderColor = this.primaryBorderColor, this.archGroupBorderWidth = "2px", this.rowOdd = "calculated", this.rowEven = "calculated", this.labelColor = "black", this.errorBkgColor = "#552222", this.errorTextColor = "#552222", this.updateColors();
  }
  updateColors() {
    var t, r, i, a, n, o, s, l, c, h, u, f, d, g, m, y, x, b, k, S, w;
    this.cScale0 = this.cScale0 || this.primaryColor, this.cScale1 = this.cScale1 || this.secondaryColor, this.cScale2 = this.cScale2 || this.tertiaryColor, this.cScale3 = this.cScale3 || v(this.primaryColor, { h: 30 }), this.cScale4 = this.cScale4 || v(this.primaryColor, { h: 60 }), this.cScale5 = this.cScale5 || v(this.primaryColor, { h: 90 }), this.cScale6 = this.cScale6 || v(this.primaryColor, { h: 120 }), this.cScale7 = this.cScale7 || v(this.primaryColor, { h: 150 }), this.cScale8 = this.cScale8 || v(this.primaryColor, { h: 210 }), this.cScale9 = this.cScale9 || v(this.primaryColor, { h: 270 }), this.cScale10 = this.cScale10 || v(this.primaryColor, { h: 300 }), this.cScale11 = this.cScale11 || v(this.primaryColor, { h: 330 }), this.cScalePeer1 = this.cScalePeer1 || X(this.secondaryColor, 45), this.cScalePeer2 = this.cScalePeer2 || X(this.tertiaryColor, 40);
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScale" + C] = X(this["cScale" + C], 10), this["cScalePeer" + C] = this["cScalePeer" + C] || X(this["cScale" + C], 25);
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScaleInv" + C] = this["cScaleInv" + C] || v(this["cScale" + C], { h: 180 });
    for (let C = 0; C < 5; C++)
      this["surface" + C] = this["surface" + C] || v(this.mainBkg, { h: 30, l: -(5 + C * 5) }), this["surfacePeer" + C] = this["surfacePeer" + C] || v(this.mainBkg, { h: 30, l: -(7 + C * 5) });
    if (this.scaleLabelColor = this.scaleLabelColor !== "calculated" && this.scaleLabelColor ? this.scaleLabelColor : this.labelTextColor, this.labelTextColor !== "calculated") {
      this.cScaleLabel0 = this.cScaleLabel0 || P(this.labelTextColor), this.cScaleLabel3 = this.cScaleLabel3 || P(this.labelTextColor);
      for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
        this["cScaleLabel" + C] = this["cScaleLabel" + C] || this.labelTextColor;
    }
    this.nodeBkg = this.mainBkg, this.nodeBorder = this.border1, this.clusterBkg = this.secondBkg, this.clusterBorder = this.border2, this.defaultLinkColor = this.lineColor, this.titleColor = this.textColor, this.edgeLabelBackground = this.labelBackground, this.actorBorder = z(this.border1, 23), this.actorBkg = this.mainBkg, this.labelBoxBkgColor = this.actorBkg, this.signalColor = this.textColor, this.signalTextColor = this.textColor, this.labelBoxBorderColor = this.actorBorder, this.labelTextColor = this.actorTextColor, this.loopTextColor = this.actorTextColor, this.noteBorderColor = this.border2, this.noteTextColor = this.actorTextColor, this.actorLineColor = this.actorBorder, this.taskTextColor = this.taskTextLightColor, this.taskTextOutsideColor = this.taskTextDarkColor, this.archEdgeColor = this.lineColor, this.archEdgeArrowColor = this.lineColor, this.rowOdd = this.rowOdd || z(this.primaryColor, 75) || "#ffffff", this.rowEven = this.rowEven || z(this.primaryColor, 1), this.transitionColor = this.transitionColor || this.lineColor, this.transitionLabelColor = this.transitionLabelColor || this.textColor, this.stateLabelColor = this.stateLabelColor || this.stateBkg || this.primaryTextColor, this.stateBkg = this.stateBkg || this.mainBkg, this.labelBackgroundColor = this.labelBackgroundColor || this.stateBkg, this.compositeBackground = this.compositeBackground || this.background || this.tertiaryColor, this.altBackground = this.altBackground || "#f0f0f0", this.compositeTitleBackground = this.compositeTitleBackground || this.mainBkg, this.compositeBorder = this.compositeBorder || this.nodeBorder, this.innerEndBackground = this.nodeBorder, this.specialStateColor = this.lineColor, this.errorBkgColor = this.errorBkgColor || this.tertiaryColor, this.errorTextColor = this.errorTextColor || this.tertiaryTextColor, this.transitionColor = this.transitionColor || this.lineColor, this.classText = this.primaryTextColor, this.fillType0 = this.primaryColor, this.fillType1 = this.secondaryColor, this.fillType2 = v(this.primaryColor, { h: 64 }), this.fillType3 = v(this.secondaryColor, { h: 64 }), this.fillType4 = v(this.primaryColor, { h: -64 }), this.fillType5 = v(this.secondaryColor, { h: -64 }), this.fillType6 = v(this.primaryColor, { h: 128 }), this.fillType7 = v(this.secondaryColor, { h: 128 }), this.pie1 = this.pie1 || this.primaryColor, this.pie2 = this.pie2 || this.secondaryColor, this.pie3 = this.pie3 || v(this.tertiaryColor, { l: -40 }), this.pie4 = this.pie4 || v(this.primaryColor, { l: -10 }), this.pie5 = this.pie5 || v(this.secondaryColor, { l: -30 }), this.pie6 = this.pie6 || v(this.tertiaryColor, { l: -20 }), this.pie7 = this.pie7 || v(this.primaryColor, { h: 60, l: -20 }), this.pie8 = this.pie8 || v(this.primaryColor, { h: -60, l: -40 }), this.pie9 = this.pie9 || v(this.primaryColor, { h: 120, l: -40 }), this.pie10 = this.pie10 || v(this.primaryColor, { h: 60, l: -40 }), this.pie11 = this.pie11 || v(this.primaryColor, { h: -90, l: -40 }), this.pie12 = this.pie12 || v(this.primaryColor, { h: 120, l: -30 }), this.pieTitleTextSize = this.pieTitleTextSize || "25px", this.pieTitleTextColor = this.pieTitleTextColor || this.taskTextDarkColor, this.pieSectionTextSize = this.pieSectionTextSize || "17px", this.pieSectionTextColor = this.pieSectionTextColor || this.textColor, this.pieLegendTextSize = this.pieLegendTextSize || "17px", this.pieLegendTextColor = this.pieLegendTextColor || this.taskTextDarkColor, this.pieStrokeColor = this.pieStrokeColor || "black", this.pieStrokeWidth = this.pieStrokeWidth || "2px", this.pieOuterStrokeWidth = this.pieOuterStrokeWidth || "2px", this.pieOuterStrokeColor = this.pieOuterStrokeColor || "black", this.pieOpacity = this.pieOpacity || "0.7", this.quadrant1Fill = this.quadrant1Fill || this.primaryColor, this.quadrant2Fill = this.quadrant2Fill || v(this.primaryColor, { r: 5, g: 5, b: 5 }), this.quadrant3Fill = this.quadrant3Fill || v(this.primaryColor, { r: 10, g: 10, b: 10 }), this.quadrant4Fill = this.quadrant4Fill || v(this.primaryColor, { r: 15, g: 15, b: 15 }), this.quadrant1TextFill = this.quadrant1TextFill || this.primaryTextColor, this.quadrant2TextFill = this.quadrant2TextFill || v(this.primaryTextColor, { r: -5, g: -5, b: -5 }), this.quadrant3TextFill = this.quadrant3TextFill || v(this.primaryTextColor, { r: -10, g: -10, b: -10 }), this.quadrant4TextFill = this.quadrant4TextFill || v(this.primaryTextColor, { r: -15, g: -15, b: -15 }), this.quadrantPointFill = this.quadrantPointFill || ci(this.quadrant1Fill) ? z(this.quadrant1Fill) : X(this.quadrant1Fill), this.quadrantPointTextFill = this.quadrantPointTextFill || this.primaryTextColor, this.quadrantXAxisTextFill = this.quadrantXAxisTextFill || this.primaryTextColor, this.quadrantYAxisTextFill = this.quadrantYAxisTextFill || this.primaryTextColor, this.quadrantInternalBorderStrokeFill = this.quadrantInternalBorderStrokeFill || this.primaryBorderColor, this.quadrantExternalBorderStrokeFill = this.quadrantExternalBorderStrokeFill || this.primaryBorderColor, this.quadrantTitleFill = this.quadrantTitleFill || this.primaryTextColor, this.radar = {
      axisColor: ((t = this.radar) == null ? void 0 : t.axisColor) || this.lineColor,
      axisStrokeWidth: ((r = this.radar) == null ? void 0 : r.axisStrokeWidth) || 2,
      axisLabelFontSize: ((i = this.radar) == null ? void 0 : i.axisLabelFontSize) || 12,
      curveOpacity: ((a = this.radar) == null ? void 0 : a.curveOpacity) || 0.5,
      curveStrokeWidth: ((n = this.radar) == null ? void 0 : n.curveStrokeWidth) || 2,
      graticuleColor: ((o = this.radar) == null ? void 0 : o.graticuleColor) || "#DEDEDE",
      graticuleStrokeWidth: ((s = this.radar) == null ? void 0 : s.graticuleStrokeWidth) || 1,
      graticuleOpacity: ((l = this.radar) == null ? void 0 : l.graticuleOpacity) || 0.3,
      legendBoxSize: ((c = this.radar) == null ? void 0 : c.legendBoxSize) || 12,
      legendFontSize: ((h = this.radar) == null ? void 0 : h.legendFontSize) || 12
    }, this.xyChart = {
      backgroundColor: ((u = this.xyChart) == null ? void 0 : u.backgroundColor) || this.background,
      titleColor: ((f = this.xyChart) == null ? void 0 : f.titleColor) || this.primaryTextColor,
      xAxisTitleColor: ((d = this.xyChart) == null ? void 0 : d.xAxisTitleColor) || this.primaryTextColor,
      xAxisLabelColor: ((g = this.xyChart) == null ? void 0 : g.xAxisLabelColor) || this.primaryTextColor,
      xAxisTickColor: ((m = this.xyChart) == null ? void 0 : m.xAxisTickColor) || this.primaryTextColor,
      xAxisLineColor: ((y = this.xyChart) == null ? void 0 : y.xAxisLineColor) || this.primaryTextColor,
      yAxisTitleColor: ((x = this.xyChart) == null ? void 0 : x.yAxisTitleColor) || this.primaryTextColor,
      yAxisLabelColor: ((b = this.xyChart) == null ? void 0 : b.yAxisLabelColor) || this.primaryTextColor,
      yAxisTickColor: ((k = this.xyChart) == null ? void 0 : k.yAxisTickColor) || this.primaryTextColor,
      yAxisLineColor: ((S = this.xyChart) == null ? void 0 : S.yAxisLineColor) || this.primaryTextColor,
      plotColorPalette: ((w = this.xyChart) == null ? void 0 : w.plotColorPalette) || "#ECECFF,#8493A6,#FFC3A0,#DCDDE1,#B8E994,#D1A36F,#C3CDE6,#FFB6C1,#496078,#F8F3E3"
    }, this.requirementBackground = this.requirementBackground || this.primaryColor, this.requirementBorderColor = this.requirementBorderColor || this.primaryBorderColor, this.requirementBorderSize = this.requirementBorderSize || "1", this.requirementTextColor = this.requirementTextColor || this.primaryTextColor, this.relationColor = this.relationColor || this.lineColor, this.relationLabelBackground = this.relationLabelBackground || this.labelBackground, this.relationLabelColor = this.relationLabelColor || this.actorTextColor, this.git0 = this.git0 || this.primaryColor, this.git1 = this.git1 || this.secondaryColor, this.git2 = this.git2 || this.tertiaryColor, this.git3 = this.git3 || v(this.primaryColor, { h: -30 }), this.git4 = this.git4 || v(this.primaryColor, { h: -60 }), this.git5 = this.git5 || v(this.primaryColor, { h: -90 }), this.git6 = this.git6 || v(this.primaryColor, { h: 60 }), this.git7 = this.git7 || v(this.primaryColor, { h: 120 }), this.darkMode ? (this.git0 = z(this.git0, 25), this.git1 = z(this.git1, 25), this.git2 = z(this.git2, 25), this.git3 = z(this.git3, 25), this.git4 = z(this.git4, 25), this.git5 = z(this.git5, 25), this.git6 = z(this.git6, 25), this.git7 = z(this.git7, 25)) : (this.git0 = X(this.git0, 25), this.git1 = X(this.git1, 25), this.git2 = X(this.git2, 25), this.git3 = X(this.git3, 25), this.git4 = X(this.git4, 25), this.git5 = X(this.git5, 25), this.git6 = X(this.git6, 25), this.git7 = X(this.git7, 25)), this.gitInv0 = this.gitInv0 || X(P(this.git0), 25), this.gitInv1 = this.gitInv1 || P(this.git1), this.gitInv2 = this.gitInv2 || P(this.git2), this.gitInv3 = this.gitInv3 || P(this.git3), this.gitInv4 = this.gitInv4 || P(this.git4), this.gitInv5 = this.gitInv5 || P(this.git5), this.gitInv6 = this.gitInv6 || P(this.git6), this.gitInv7 = this.gitInv7 || P(this.git7), this.gitBranchLabel0 = this.gitBranchLabel0 || P(this.labelTextColor), this.gitBranchLabel1 = this.gitBranchLabel1 || this.labelTextColor, this.gitBranchLabel2 = this.gitBranchLabel2 || this.labelTextColor, this.gitBranchLabel3 = this.gitBranchLabel3 || P(this.labelTextColor), this.gitBranchLabel4 = this.gitBranchLabel4 || this.labelTextColor, this.gitBranchLabel5 = this.gitBranchLabel5 || this.labelTextColor, this.gitBranchLabel6 = this.gitBranchLabel6 || this.labelTextColor, this.gitBranchLabel7 = this.gitBranchLabel7 || this.labelTextColor, this.tagLabelColor = this.tagLabelColor || this.primaryTextColor, this.tagLabelBackground = this.tagLabelBackground || this.primaryColor, this.tagLabelBorder = this.tagBorder || this.primaryBorderColor, this.tagLabelFontSize = this.tagLabelFontSize || "10px", this.commitLabelColor = this.commitLabelColor || this.secondaryTextColor, this.commitLabelBackground = this.commitLabelBackground || this.secondaryColor, this.commitLabelFontSize = this.commitLabelFontSize || "10px", this.attributeBackgroundColorOdd = this.attributeBackgroundColorOdd || Sa, this.attributeBackgroundColorEven = this.attributeBackgroundColorEven || Ta;
  }
  calculate(t) {
    if (Object.keys(this).forEach((i) => {
      this[i] === "calculated" && (this[i] = void 0);
    }), typeof t != "object") {
      this.updateColors();
      return;
    }
    const r = Object.keys(t);
    r.forEach((i) => {
      this[i] = t[i];
    }), this.updateColors(), r.forEach((i) => {
      this[i] = t[i];
    });
  }
}, p(cr, "Theme"), cr), mg = /* @__PURE__ */ p((e) => {
  const t = new gg();
  return t.calculate(e), t;
}, "getThemeVariables"), hr, yg = (hr = class {
  constructor() {
    this.background = "#f4f4f4", this.primaryColor = "#cde498", this.secondaryColor = "#cdffb2", this.background = "white", this.mainBkg = "#cde498", this.secondBkg = "#cdffb2", this.lineColor = "green", this.border1 = "#13540c", this.border2 = "#6eaa49", this.arrowheadColor = "green", this.fontFamily = '"trebuchet ms", verdana, arial, sans-serif', this.fontSize = "16px", this.tertiaryColor = z("#cde498", 10), this.primaryBorderColor = $t(this.primaryColor, this.darkMode), this.secondaryBorderColor = $t(this.secondaryColor, this.darkMode), this.tertiaryBorderColor = $t(this.tertiaryColor, this.darkMode), this.primaryTextColor = P(this.primaryColor), this.secondaryTextColor = P(this.secondaryColor), this.tertiaryTextColor = P(this.primaryColor), this.lineColor = P(this.background), this.textColor = P(this.background), this.THEME_COLOR_LIMIT = 12, this.nodeBkg = "calculated", this.nodeBorder = "calculated", this.clusterBkg = "calculated", this.clusterBorder = "calculated", this.defaultLinkColor = "calculated", this.titleColor = "#333", this.edgeLabelBackground = "#e8e8e8", this.actorBorder = "calculated", this.actorBkg = "calculated", this.actorTextColor = "black", this.actorLineColor = "calculated", this.signalColor = "#333", this.signalTextColor = "#333", this.labelBoxBkgColor = "calculated", this.labelBoxBorderColor = "#326932", this.labelTextColor = "calculated", this.loopTextColor = "calculated", this.noteBorderColor = "calculated", this.noteBkgColor = "#fff5ad", this.noteTextColor = "calculated", this.activationBorderColor = "#666", this.activationBkgColor = "#f4f4f4", this.sequenceNumberColor = "white", this.sectionBkgColor = "#6eaa49", this.altSectionBkgColor = "white", this.sectionBkgColor2 = "#6eaa49", this.excludeBkgColor = "#eeeeee", this.taskBorderColor = "calculated", this.taskBkgColor = "#487e3a", this.taskTextLightColor = "white", this.taskTextColor = "calculated", this.taskTextDarkColor = "black", this.taskTextOutsideColor = "calculated", this.taskTextClickableColor = "#003163", this.activeTaskBorderColor = "calculated", this.activeTaskBkgColor = "calculated", this.gridColor = "lightgrey", this.doneTaskBkgColor = "lightgrey", this.doneTaskBorderColor = "grey", this.critBorderColor = "#ff8888", this.critBkgColor = "red", this.todayLineColor = "red", this.vertLineColor = "#00BFFF", this.personBorder = this.primaryBorderColor, this.personBkg = this.mainBkg, this.archEdgeColor = "calculated", this.archEdgeArrowColor = "calculated", this.archEdgeWidth = "3", this.archGroupBorderColor = this.primaryBorderColor, this.archGroupBorderWidth = "2px", this.labelColor = "black", this.errorBkgColor = "#552222", this.errorTextColor = "#552222";
  }
  updateColors() {
    var t, r, i, a, n, o, s, l, c, h, u, f, d, g, m, y, x, b, k, S, w;
    this.actorBorder = X(this.mainBkg, 20), this.actorBkg = this.mainBkg, this.labelBoxBkgColor = this.actorBkg, this.labelTextColor = this.actorTextColor, this.loopTextColor = this.actorTextColor, this.noteBorderColor = this.border2, this.noteTextColor = this.actorTextColor, this.actorLineColor = this.actorBorder, this.cScale0 = this.cScale0 || this.primaryColor, this.cScale1 = this.cScale1 || this.secondaryColor, this.cScale2 = this.cScale2 || this.tertiaryColor, this.cScale3 = this.cScale3 || v(this.primaryColor, { h: 30 }), this.cScale4 = this.cScale4 || v(this.primaryColor, { h: 60 }), this.cScale5 = this.cScale5 || v(this.primaryColor, { h: 90 }), this.cScale6 = this.cScale6 || v(this.primaryColor, { h: 120 }), this.cScale7 = this.cScale7 || v(this.primaryColor, { h: 150 }), this.cScale8 = this.cScale8 || v(this.primaryColor, { h: 210 }), this.cScale9 = this.cScale9 || v(this.primaryColor, { h: 270 }), this.cScale10 = this.cScale10 || v(this.primaryColor, { h: 300 }), this.cScale11 = this.cScale11 || v(this.primaryColor, { h: 330 }), this.cScalePeer1 = this.cScalePeer1 || X(this.secondaryColor, 45), this.cScalePeer2 = this.cScalePeer2 || X(this.tertiaryColor, 40);
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScale" + C] = X(this["cScale" + C], 10), this["cScalePeer" + C] = this["cScalePeer" + C] || X(this["cScale" + C], 25);
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScaleInv" + C] = this["cScaleInv" + C] || v(this["cScale" + C], { h: 180 });
    this.scaleLabelColor = this.scaleLabelColor !== "calculated" && this.scaleLabelColor ? this.scaleLabelColor : this.labelTextColor;
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScaleLabel" + C] = this["cScaleLabel" + C] || this.scaleLabelColor;
    for (let C = 0; C < 5; C++)
      this["surface" + C] = this["surface" + C] || v(this.mainBkg, { h: 30, s: -30, l: -(5 + C * 5) }), this["surfacePeer" + C] = this["surfacePeer" + C] || v(this.mainBkg, { h: 30, s: -30, l: -(8 + C * 5) });
    this.nodeBkg = this.mainBkg, this.nodeBorder = this.border1, this.clusterBkg = this.secondBkg, this.clusterBorder = this.border2, this.defaultLinkColor = this.lineColor, this.taskBorderColor = this.border1, this.taskTextColor = this.taskTextLightColor, this.taskTextOutsideColor = this.taskTextDarkColor, this.activeTaskBorderColor = this.taskBorderColor, this.activeTaskBkgColor = this.mainBkg, this.archEdgeColor = this.lineColor, this.archEdgeArrowColor = this.lineColor, this.rowOdd = this.rowOdd || z(this.mainBkg, 75) || "#ffffff", this.rowEven = this.rowEven || z(this.mainBkg, 20), this.transitionColor = this.transitionColor || this.lineColor, this.transitionLabelColor = this.transitionLabelColor || this.textColor, this.stateLabelColor = this.stateLabelColor || this.stateBkg || this.primaryTextColor, this.stateBkg = this.stateBkg || this.mainBkg, this.labelBackgroundColor = this.labelBackgroundColor || this.stateBkg, this.compositeBackground = this.compositeBackground || this.background || this.tertiaryColor, this.altBackground = this.altBackground || "#f0f0f0", this.compositeTitleBackground = this.compositeTitleBackground || this.mainBkg, this.compositeBorder = this.compositeBorder || this.nodeBorder, this.innerEndBackground = this.primaryBorderColor, this.specialStateColor = this.lineColor, this.errorBkgColor = this.errorBkgColor || this.tertiaryColor, this.errorTextColor = this.errorTextColor || this.tertiaryTextColor, this.transitionColor = this.transitionColor || this.lineColor, this.classText = this.primaryTextColor, this.fillType0 = this.primaryColor, this.fillType1 = this.secondaryColor, this.fillType2 = v(this.primaryColor, { h: 64 }), this.fillType3 = v(this.secondaryColor, { h: 64 }), this.fillType4 = v(this.primaryColor, { h: -64 }), this.fillType5 = v(this.secondaryColor, { h: -64 }), this.fillType6 = v(this.primaryColor, { h: 128 }), this.fillType7 = v(this.secondaryColor, { h: 128 }), this.pie1 = this.pie1 || this.primaryColor, this.pie2 = this.pie2 || this.secondaryColor, this.pie3 = this.pie3 || this.tertiaryColor, this.pie4 = this.pie4 || v(this.primaryColor, { l: -30 }), this.pie5 = this.pie5 || v(this.secondaryColor, { l: -30 }), this.pie6 = this.pie6 || v(this.tertiaryColor, { h: 40, l: -40 }), this.pie7 = this.pie7 || v(this.primaryColor, { h: 60, l: -10 }), this.pie8 = this.pie8 || v(this.primaryColor, { h: -60, l: -10 }), this.pie9 = this.pie9 || v(this.primaryColor, { h: 120, l: 0 }), this.pie10 = this.pie10 || v(this.primaryColor, { h: 60, l: -50 }), this.pie11 = this.pie11 || v(this.primaryColor, { h: -60, l: -50 }), this.pie12 = this.pie12 || v(this.primaryColor, { h: 120, l: -50 }), this.pieTitleTextSize = this.pieTitleTextSize || "25px", this.pieTitleTextColor = this.pieTitleTextColor || this.taskTextDarkColor, this.pieSectionTextSize = this.pieSectionTextSize || "17px", this.pieSectionTextColor = this.pieSectionTextColor || this.textColor, this.pieLegendTextSize = this.pieLegendTextSize || "17px", this.pieLegendTextColor = this.pieLegendTextColor || this.taskTextDarkColor, this.pieStrokeColor = this.pieStrokeColor || "black", this.pieStrokeWidth = this.pieStrokeWidth || "2px", this.pieOuterStrokeWidth = this.pieOuterStrokeWidth || "2px", this.pieOuterStrokeColor = this.pieOuterStrokeColor || "black", this.pieOpacity = this.pieOpacity || "0.7", this.quadrant1Fill = this.quadrant1Fill || this.primaryColor, this.quadrant2Fill = this.quadrant2Fill || v(this.primaryColor, { r: 5, g: 5, b: 5 }), this.quadrant3Fill = this.quadrant3Fill || v(this.primaryColor, { r: 10, g: 10, b: 10 }), this.quadrant4Fill = this.quadrant4Fill || v(this.primaryColor, { r: 15, g: 15, b: 15 }), this.quadrant1TextFill = this.quadrant1TextFill || this.primaryTextColor, this.quadrant2TextFill = this.quadrant2TextFill || v(this.primaryTextColor, { r: -5, g: -5, b: -5 }), this.quadrant3TextFill = this.quadrant3TextFill || v(this.primaryTextColor, { r: -10, g: -10, b: -10 }), this.quadrant4TextFill = this.quadrant4TextFill || v(this.primaryTextColor, { r: -15, g: -15, b: -15 }), this.quadrantPointFill = this.quadrantPointFill || ci(this.quadrant1Fill) ? z(this.quadrant1Fill) : X(this.quadrant1Fill), this.quadrantPointTextFill = this.quadrantPointTextFill || this.primaryTextColor, this.quadrantXAxisTextFill = this.quadrantXAxisTextFill || this.primaryTextColor, this.quadrantYAxisTextFill = this.quadrantYAxisTextFill || this.primaryTextColor, this.quadrantInternalBorderStrokeFill = this.quadrantInternalBorderStrokeFill || this.primaryBorderColor, this.quadrantExternalBorderStrokeFill = this.quadrantExternalBorderStrokeFill || this.primaryBorderColor, this.quadrantTitleFill = this.quadrantTitleFill || this.primaryTextColor, this.packet = {
      startByteColor: this.primaryTextColor,
      endByteColor: this.primaryTextColor,
      labelColor: this.primaryTextColor,
      titleColor: this.primaryTextColor,
      blockStrokeColor: this.primaryTextColor,
      blockFillColor: this.mainBkg
    }, this.radar = {
      axisColor: ((t = this.radar) == null ? void 0 : t.axisColor) || this.lineColor,
      axisStrokeWidth: ((r = this.radar) == null ? void 0 : r.axisStrokeWidth) || 2,
      axisLabelFontSize: ((i = this.radar) == null ? void 0 : i.axisLabelFontSize) || 12,
      curveOpacity: ((a = this.radar) == null ? void 0 : a.curveOpacity) || 0.5,
      curveStrokeWidth: ((n = this.radar) == null ? void 0 : n.curveStrokeWidth) || 2,
      graticuleColor: ((o = this.radar) == null ? void 0 : o.graticuleColor) || "#DEDEDE",
      graticuleStrokeWidth: ((s = this.radar) == null ? void 0 : s.graticuleStrokeWidth) || 1,
      graticuleOpacity: ((l = this.radar) == null ? void 0 : l.graticuleOpacity) || 0.3,
      legendBoxSize: ((c = this.radar) == null ? void 0 : c.legendBoxSize) || 12,
      legendFontSize: ((h = this.radar) == null ? void 0 : h.legendFontSize) || 12
    }, this.xyChart = {
      backgroundColor: ((u = this.xyChart) == null ? void 0 : u.backgroundColor) || this.background,
      titleColor: ((f = this.xyChart) == null ? void 0 : f.titleColor) || this.primaryTextColor,
      xAxisTitleColor: ((d = this.xyChart) == null ? void 0 : d.xAxisTitleColor) || this.primaryTextColor,
      xAxisLabelColor: ((g = this.xyChart) == null ? void 0 : g.xAxisLabelColor) || this.primaryTextColor,
      xAxisTickColor: ((m = this.xyChart) == null ? void 0 : m.xAxisTickColor) || this.primaryTextColor,
      xAxisLineColor: ((y = this.xyChart) == null ? void 0 : y.xAxisLineColor) || this.primaryTextColor,
      yAxisTitleColor: ((x = this.xyChart) == null ? void 0 : x.yAxisTitleColor) || this.primaryTextColor,
      yAxisLabelColor: ((b = this.xyChart) == null ? void 0 : b.yAxisLabelColor) || this.primaryTextColor,
      yAxisTickColor: ((k = this.xyChart) == null ? void 0 : k.yAxisTickColor) || this.primaryTextColor,
      yAxisLineColor: ((S = this.xyChart) == null ? void 0 : S.yAxisLineColor) || this.primaryTextColor,
      plotColorPalette: ((w = this.xyChart) == null ? void 0 : w.plotColorPalette) || "#CDE498,#FF6B6B,#A0D2DB,#D7BDE2,#F0F0F0,#FFC3A0,#7FD8BE,#FF9A8B,#FAF3E0,#FFF176"
    }, this.requirementBackground = this.requirementBackground || this.primaryColor, this.requirementBorderColor = this.requirementBorderColor || this.primaryBorderColor, this.requirementBorderSize = this.requirementBorderSize || "1", this.requirementTextColor = this.requirementTextColor || this.primaryTextColor, this.relationColor = this.relationColor || this.lineColor, this.relationLabelBackground = this.relationLabelBackground || this.edgeLabelBackground, this.relationLabelColor = this.relationLabelColor || this.actorTextColor, this.git0 = this.git0 || this.primaryColor, this.git1 = this.git1 || this.secondaryColor, this.git2 = this.git2 || this.tertiaryColor, this.git3 = this.git3 || v(this.primaryColor, { h: -30 }), this.git4 = this.git4 || v(this.primaryColor, { h: -60 }), this.git5 = this.git5 || v(this.primaryColor, { h: -90 }), this.git6 = this.git6 || v(this.primaryColor, { h: 60 }), this.git7 = this.git7 || v(this.primaryColor, { h: 120 }), this.darkMode ? (this.git0 = z(this.git0, 25), this.git1 = z(this.git1, 25), this.git2 = z(this.git2, 25), this.git3 = z(this.git3, 25), this.git4 = z(this.git4, 25), this.git5 = z(this.git5, 25), this.git6 = z(this.git6, 25), this.git7 = z(this.git7, 25)) : (this.git0 = X(this.git0, 25), this.git1 = X(this.git1, 25), this.git2 = X(this.git2, 25), this.git3 = X(this.git3, 25), this.git4 = X(this.git4, 25), this.git5 = X(this.git5, 25), this.git6 = X(this.git6, 25), this.git7 = X(this.git7, 25)), this.gitInv0 = this.gitInv0 || P(this.git0), this.gitInv1 = this.gitInv1 || P(this.git1), this.gitInv2 = this.gitInv2 || P(this.git2), this.gitInv3 = this.gitInv3 || P(this.git3), this.gitInv4 = this.gitInv4 || P(this.git4), this.gitInv5 = this.gitInv5 || P(this.git5), this.gitInv6 = this.gitInv6 || P(this.git6), this.gitInv7 = this.gitInv7 || P(this.git7), this.gitBranchLabel0 = this.gitBranchLabel0 || P(this.labelTextColor), this.gitBranchLabel1 = this.gitBranchLabel1 || this.labelTextColor, this.gitBranchLabel2 = this.gitBranchLabel2 || this.labelTextColor, this.gitBranchLabel3 = this.gitBranchLabel3 || P(this.labelTextColor), this.gitBranchLabel4 = this.gitBranchLabel4 || this.labelTextColor, this.gitBranchLabel5 = this.gitBranchLabel5 || this.labelTextColor, this.gitBranchLabel6 = this.gitBranchLabel6 || this.labelTextColor, this.gitBranchLabel7 = this.gitBranchLabel7 || this.labelTextColor, this.tagLabelColor = this.tagLabelColor || this.primaryTextColor, this.tagLabelBackground = this.tagLabelBackground || this.primaryColor, this.tagLabelBorder = this.tagBorder || this.primaryBorderColor, this.tagLabelFontSize = this.tagLabelFontSize || "10px", this.commitLabelColor = this.commitLabelColor || this.secondaryTextColor, this.commitLabelBackground = this.commitLabelBackground || this.secondaryColor, this.commitLabelFontSize = this.commitLabelFontSize || "10px", this.attributeBackgroundColorOdd = this.attributeBackgroundColorOdd || Sa, this.attributeBackgroundColorEven = this.attributeBackgroundColorEven || Ta;
  }
  calculate(t) {
    if (typeof t != "object") {
      this.updateColors();
      return;
    }
    const r = Object.keys(t);
    r.forEach((i) => {
      this[i] = t[i];
    }), this.updateColors(), r.forEach((i) => {
      this[i] = t[i];
    });
  }
}, p(hr, "Theme"), hr), xg = /* @__PURE__ */ p((e) => {
  const t = new yg();
  return t.calculate(e), t;
}, "getThemeVariables"), ur, bg = (ur = class {
  constructor() {
    this.primaryColor = "#eee", this.contrast = "#707070", this.secondaryColor = z(this.contrast, 55), this.background = "#ffffff", this.tertiaryColor = v(this.primaryColor, { h: -160 }), this.primaryBorderColor = $t(this.primaryColor, this.darkMode), this.secondaryBorderColor = $t(this.secondaryColor, this.darkMode), this.tertiaryBorderColor = $t(this.tertiaryColor, this.darkMode), this.primaryTextColor = P(this.primaryColor), this.secondaryTextColor = P(this.secondaryColor), this.tertiaryTextColor = P(this.tertiaryColor), this.lineColor = P(this.background), this.textColor = P(this.background), this.mainBkg = "#eee", this.secondBkg = "calculated", this.lineColor = "#666", this.border1 = "#999", this.border2 = "calculated", this.note = "#ffa", this.text = "#333", this.critical = "#d42", this.done = "#bbb", this.arrowheadColor = "#333333", this.fontFamily = '"trebuchet ms", verdana, arial, sans-serif', this.fontSize = "16px", this.THEME_COLOR_LIMIT = 12, this.nodeBkg = "calculated", this.nodeBorder = "calculated", this.clusterBkg = "calculated", this.clusterBorder = "calculated", this.defaultLinkColor = "calculated", this.titleColor = "calculated", this.edgeLabelBackground = "white", this.actorBorder = "calculated", this.actorBkg = "calculated", this.actorTextColor = "calculated", this.actorLineColor = this.actorBorder, this.signalColor = "calculated", this.signalTextColor = "calculated", this.labelBoxBkgColor = "calculated", this.labelBoxBorderColor = "calculated", this.labelTextColor = "calculated", this.loopTextColor = "calculated", this.noteBorderColor = "calculated", this.noteBkgColor = "calculated", this.noteTextColor = "calculated", this.activationBorderColor = "#666", this.activationBkgColor = "#f4f4f4", this.sequenceNumberColor = "white", this.sectionBkgColor = "calculated", this.altSectionBkgColor = "white", this.sectionBkgColor2 = "calculated", this.excludeBkgColor = "#eeeeee", this.taskBorderColor = "calculated", this.taskBkgColor = "calculated", this.taskTextLightColor = "white", this.taskTextColor = "calculated", this.taskTextDarkColor = "calculated", this.taskTextOutsideColor = "calculated", this.taskTextClickableColor = "#003163", this.activeTaskBorderColor = "calculated", this.activeTaskBkgColor = "calculated", this.gridColor = "calculated", this.doneTaskBkgColor = "calculated", this.doneTaskBorderColor = "calculated", this.critBkgColor = "calculated", this.critBorderColor = "calculated", this.todayLineColor = "calculated", this.vertLineColor = "calculated", this.personBorder = this.primaryBorderColor, this.personBkg = this.mainBkg, this.archEdgeColor = "calculated", this.archEdgeArrowColor = "calculated", this.archEdgeWidth = "3", this.archGroupBorderColor = this.primaryBorderColor, this.archGroupBorderWidth = "2px", this.rowOdd = this.rowOdd || z(this.mainBkg, 75) || "#ffffff", this.rowEven = this.rowEven || "#f4f4f4", this.labelColor = "black", this.errorBkgColor = "#552222", this.errorTextColor = "#552222";
  }
  updateColors() {
    var t, r, i, a, n, o, s, l, c, h, u, f, d, g, m, y, x, b, k, S, w;
    this.secondBkg = z(this.contrast, 55), this.border2 = this.contrast, this.actorBorder = z(this.border1, 23), this.actorBkg = this.mainBkg, this.actorTextColor = this.text, this.actorLineColor = this.actorBorder, this.signalColor = this.text, this.signalTextColor = this.text, this.labelBoxBkgColor = this.actorBkg, this.labelBoxBorderColor = this.actorBorder, this.labelTextColor = this.text, this.loopTextColor = this.text, this.noteBorderColor = "#999", this.noteBkgColor = "#666", this.noteTextColor = "#fff", this.cScale0 = this.cScale0 || "#555", this.cScale1 = this.cScale1 || "#F4F4F4", this.cScale2 = this.cScale2 || "#555", this.cScale3 = this.cScale3 || "#BBB", this.cScale4 = this.cScale4 || "#777", this.cScale5 = this.cScale5 || "#999", this.cScale6 = this.cScale6 || "#DDD", this.cScale7 = this.cScale7 || "#FFF", this.cScale8 = this.cScale8 || "#DDD", this.cScale9 = this.cScale9 || "#BBB", this.cScale10 = this.cScale10 || "#999", this.cScale11 = this.cScale11 || "#777";
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScaleInv" + C] = this["cScaleInv" + C] || P(this["cScale" + C]);
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this.darkMode ? this["cScalePeer" + C] = this["cScalePeer" + C] || z(this["cScale" + C], 10) : this["cScalePeer" + C] = this["cScalePeer" + C] || X(this["cScale" + C], 10);
    this.scaleLabelColor = this.scaleLabelColor || (this.darkMode ? "black" : this.labelTextColor), this.cScaleLabel0 = this.cScaleLabel0 || this.cScale1, this.cScaleLabel2 = this.cScaleLabel2 || this.cScale1;
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["cScaleLabel" + C] = this["cScaleLabel" + C] || this.scaleLabelColor;
    for (let C = 0; C < 5; C++)
      this["surface" + C] = this["surface" + C] || v(this.mainBkg, { l: -(5 + C * 5) }), this["surfacePeer" + C] = this["surfacePeer" + C] || v(this.mainBkg, { l: -(8 + C * 5) });
    this.nodeBkg = this.mainBkg, this.nodeBorder = this.border1, this.clusterBkg = this.secondBkg, this.clusterBorder = this.border2, this.defaultLinkColor = this.lineColor, this.titleColor = this.text, this.sectionBkgColor = z(this.contrast, 30), this.sectionBkgColor2 = z(this.contrast, 30), this.taskBorderColor = X(this.contrast, 10), this.taskBkgColor = this.contrast, this.taskTextColor = this.taskTextLightColor, this.taskTextDarkColor = this.text, this.taskTextOutsideColor = this.taskTextDarkColor, this.activeTaskBorderColor = this.taskBorderColor, this.activeTaskBkgColor = this.mainBkg, this.gridColor = z(this.border1, 30), this.doneTaskBkgColor = this.done, this.doneTaskBorderColor = this.lineColor, this.critBkgColor = this.critical, this.critBorderColor = X(this.critBkgColor, 10), this.todayLineColor = this.critBkgColor, this.vertLineColor = this.critBkgColor, this.archEdgeColor = this.lineColor, this.archEdgeArrowColor = this.lineColor, this.transitionColor = this.transitionColor || "#000", this.transitionLabelColor = this.transitionLabelColor || this.textColor, this.stateLabelColor = this.stateLabelColor || this.stateBkg || this.primaryTextColor, this.stateBkg = this.stateBkg || this.mainBkg, this.labelBackgroundColor = this.labelBackgroundColor || this.stateBkg, this.compositeBackground = this.compositeBackground || this.background || this.tertiaryColor, this.altBackground = this.altBackground || "#f4f4f4", this.compositeTitleBackground = this.compositeTitleBackground || this.mainBkg, this.stateBorder = this.stateBorder || "#000", this.innerEndBackground = this.primaryBorderColor, this.specialStateColor = "#222", this.errorBkgColor = this.errorBkgColor || this.tertiaryColor, this.errorTextColor = this.errorTextColor || this.tertiaryTextColor, this.classText = this.primaryTextColor, this.fillType0 = this.primaryColor, this.fillType1 = this.secondaryColor, this.fillType2 = v(this.primaryColor, { h: 64 }), this.fillType3 = v(this.secondaryColor, { h: 64 }), this.fillType4 = v(this.primaryColor, { h: -64 }), this.fillType5 = v(this.secondaryColor, { h: -64 }), this.fillType6 = v(this.primaryColor, { h: 128 }), this.fillType7 = v(this.secondaryColor, { h: 128 });
    for (let C = 0; C < this.THEME_COLOR_LIMIT; C++)
      this["pie" + C] = this["cScale" + C];
    this.pie12 = this.pie0, this.pieTitleTextSize = this.pieTitleTextSize || "25px", this.pieTitleTextColor = this.pieTitleTextColor || this.taskTextDarkColor, this.pieSectionTextSize = this.pieSectionTextSize || "17px", this.pieSectionTextColor = this.pieSectionTextColor || this.textColor, this.pieLegendTextSize = this.pieLegendTextSize || "17px", this.pieLegendTextColor = this.pieLegendTextColor || this.taskTextDarkColor, this.pieStrokeColor = this.pieStrokeColor || "black", this.pieStrokeWidth = this.pieStrokeWidth || "2px", this.pieOuterStrokeWidth = this.pieOuterStrokeWidth || "2px", this.pieOuterStrokeColor = this.pieOuterStrokeColor || "black", this.pieOpacity = this.pieOpacity || "0.7", this.quadrant1Fill = this.quadrant1Fill || this.primaryColor, this.quadrant2Fill = this.quadrant2Fill || v(this.primaryColor, { r: 5, g: 5, b: 5 }), this.quadrant3Fill = this.quadrant3Fill || v(this.primaryColor, { r: 10, g: 10, b: 10 }), this.quadrant4Fill = this.quadrant4Fill || v(this.primaryColor, { r: 15, g: 15, b: 15 }), this.quadrant1TextFill = this.quadrant1TextFill || this.primaryTextColor, this.quadrant2TextFill = this.quadrant2TextFill || v(this.primaryTextColor, { r: -5, g: -5, b: -5 }), this.quadrant3TextFill = this.quadrant3TextFill || v(this.primaryTextColor, { r: -10, g: -10, b: -10 }), this.quadrant4TextFill = this.quadrant4TextFill || v(this.primaryTextColor, { r: -15, g: -15, b: -15 }), this.quadrantPointFill = this.quadrantPointFill || ci(this.quadrant1Fill) ? z(this.quadrant1Fill) : X(this.quadrant1Fill), this.quadrantPointTextFill = this.quadrantPointTextFill || this.primaryTextColor, this.quadrantXAxisTextFill = this.quadrantXAxisTextFill || this.primaryTextColor, this.quadrantYAxisTextFill = this.quadrantYAxisTextFill || this.primaryTextColor, this.quadrantInternalBorderStrokeFill = this.quadrantInternalBorderStrokeFill || this.primaryBorderColor, this.quadrantExternalBorderStrokeFill = this.quadrantExternalBorderStrokeFill || this.primaryBorderColor, this.quadrantTitleFill = this.quadrantTitleFill || this.primaryTextColor, this.xyChart = {
      backgroundColor: ((t = this.xyChart) == null ? void 0 : t.backgroundColor) || this.background,
      titleColor: ((r = this.xyChart) == null ? void 0 : r.titleColor) || this.primaryTextColor,
      xAxisTitleColor: ((i = this.xyChart) == null ? void 0 : i.xAxisTitleColor) || this.primaryTextColor,
      xAxisLabelColor: ((a = this.xyChart) == null ? void 0 : a.xAxisLabelColor) || this.primaryTextColor,
      xAxisTickColor: ((n = this.xyChart) == null ? void 0 : n.xAxisTickColor) || this.primaryTextColor,
      xAxisLineColor: ((o = this.xyChart) == null ? void 0 : o.xAxisLineColor) || this.primaryTextColor,
      yAxisTitleColor: ((s = this.xyChart) == null ? void 0 : s.yAxisTitleColor) || this.primaryTextColor,
      yAxisLabelColor: ((l = this.xyChart) == null ? void 0 : l.yAxisLabelColor) || this.primaryTextColor,
      yAxisTickColor: ((c = this.xyChart) == null ? void 0 : c.yAxisTickColor) || this.primaryTextColor,
      yAxisLineColor: ((h = this.xyChart) == null ? void 0 : h.yAxisLineColor) || this.primaryTextColor,
      plotColorPalette: ((u = this.xyChart) == null ? void 0 : u.plotColorPalette) || "#EEE,#6BB8E4,#8ACB88,#C7ACD6,#E8DCC2,#FFB2A8,#FFF380,#7E8D91,#FFD8B1,#FAF3E0"
    }, this.radar = {
      axisColor: ((f = this.radar) == null ? void 0 : f.axisColor) || this.lineColor,
      axisStrokeWidth: ((d = this.radar) == null ? void 0 : d.axisStrokeWidth) || 2,
      axisLabelFontSize: ((g = this.radar) == null ? void 0 : g.axisLabelFontSize) || 12,
      curveOpacity: ((m = this.radar) == null ? void 0 : m.curveOpacity) || 0.5,
      curveStrokeWidth: ((y = this.radar) == null ? void 0 : y.curveStrokeWidth) || 2,
      graticuleColor: ((x = this.radar) == null ? void 0 : x.graticuleColor) || "#DEDEDE",
      graticuleStrokeWidth: ((b = this.radar) == null ? void 0 : b.graticuleStrokeWidth) || 1,
      graticuleOpacity: ((k = this.radar) == null ? void 0 : k.graticuleOpacity) || 0.3,
      legendBoxSize: ((S = this.radar) == null ? void 0 : S.legendBoxSize) || 12,
      legendFontSize: ((w = this.radar) == null ? void 0 : w.legendFontSize) || 12
    }, this.requirementBackground = this.requirementBackground || this.primaryColor, this.requirementBorderColor = this.requirementBorderColor || this.primaryBorderColor, this.requirementBorderSize = this.requirementBorderSize || "1", this.requirementTextColor = this.requirementTextColor || this.primaryTextColor, this.relationColor = this.relationColor || this.lineColor, this.relationLabelBackground = this.relationLabelBackground || this.edgeLabelBackground, this.relationLabelColor = this.relationLabelColor || this.actorTextColor, this.git0 = X(this.pie1, 25) || this.primaryColor, this.git1 = this.pie2 || this.secondaryColor, this.git2 = this.pie3 || this.tertiaryColor, this.git3 = this.pie4 || v(this.primaryColor, { h: -30 }), this.git4 = this.pie5 || v(this.primaryColor, { h: -60 }), this.git5 = this.pie6 || v(this.primaryColor, { h: -90 }), this.git6 = this.pie7 || v(this.primaryColor, { h: 60 }), this.git7 = this.pie8 || v(this.primaryColor, { h: 120 }), this.gitInv0 = this.gitInv0 || P(this.git0), this.gitInv1 = this.gitInv1 || P(this.git1), this.gitInv2 = this.gitInv2 || P(this.git2), this.gitInv3 = this.gitInv3 || P(this.git3), this.gitInv4 = this.gitInv4 || P(this.git4), this.gitInv5 = this.gitInv5 || P(this.git5), this.gitInv6 = this.gitInv6 || P(this.git6), this.gitInv7 = this.gitInv7 || P(this.git7), this.branchLabelColor = this.branchLabelColor || this.labelTextColor, this.gitBranchLabel0 = this.branchLabelColor, this.gitBranchLabel1 = "white", this.gitBranchLabel2 = this.branchLabelColor, this.gitBranchLabel3 = "white", this.gitBranchLabel4 = this.branchLabelColor, this.gitBranchLabel5 = this.branchLabelColor, this.gitBranchLabel6 = this.branchLabelColor, this.gitBranchLabel7 = this.branchLabelColor, this.tagLabelColor = this.tagLabelColor || this.primaryTextColor, this.tagLabelBackground = this.tagLabelBackground || this.primaryColor, this.tagLabelBorder = this.tagBorder || this.primaryBorderColor, this.tagLabelFontSize = this.tagLabelFontSize || "10px", this.commitLabelColor = this.commitLabelColor || this.secondaryTextColor, this.commitLabelBackground = this.commitLabelBackground || this.secondaryColor, this.commitLabelFontSize = this.commitLabelFontSize || "10px", this.attributeBackgroundColorOdd = this.attributeBackgroundColorOdd || Sa, this.attributeBackgroundColorEven = this.attributeBackgroundColorEven || Ta;
  }
  calculate(t) {
    if (typeof t != "object") {
      this.updateColors();
      return;
    }
    const r = Object.keys(t);
    r.forEach((i) => {
      this[i] = t[i];
    }), this.updateColors(), r.forEach((i) => {
      this[i] = t[i];
    });
  }
}, p(ur, "Theme"), ur), Cg = /* @__PURE__ */ p((e) => {
  const t = new bg();
  return t.calculate(e), t;
}, "getThemeVariables"), ue = {
  base: {
    getThemeVariables: fg
  },
  dark: {
    getThemeVariables: pg
  },
  default: {
    getThemeVariables: mg
  },
  forest: {
    getThemeVariables: xg
  },
  neutral: {
    getThemeVariables: Cg
  }
}, Vt = {
  flowchart: {
    useMaxWidth: !0,
    titleTopMargin: 25,
    subGraphTitleMargin: {
      top: 0,
      bottom: 0
    },
    diagramPadding: 8,
    htmlLabels: !0,
    nodeSpacing: 50,
    rankSpacing: 50,
    curve: "basis",
    padding: 15,
    defaultRenderer: "dagre-wrapper",
    wrappingWidth: 200,
    inheritDir: !1
  },
  sequence: {
    useMaxWidth: !0,
    hideUnusedParticipants: !1,
    activationWidth: 10,
    diagramMarginX: 50,
    diagramMarginY: 10,
    actorMargin: 50,
    width: 150,
    height: 65,
    boxMargin: 10,
    boxTextMargin: 5,
    noteMargin: 10,
    messageMargin: 35,
    messageAlign: "center",
    mirrorActors: !0,
    forceMenus: !1,
    bottomMarginAdj: 1,
    rightAngles: !1,
    showSequenceNumbers: !1,
    actorFontSize: 14,
    actorFontFamily: '"Open Sans", sans-serif',
    actorFontWeight: 400,
    noteFontSize: 14,
    noteFontFamily: '"trebuchet ms", verdana, arial, sans-serif',
    noteFontWeight: 400,
    noteAlign: "center",
    messageFontSize: 16,
    messageFontFamily: '"trebuchet ms", verdana, arial, sans-serif',
    messageFontWeight: 400,
    wrap: !1,
    wrapPadding: 10,
    labelBoxWidth: 50,
    labelBoxHeight: 20
  },
  gantt: {
    useMaxWidth: !0,
    titleTopMargin: 25,
    barHeight: 20,
    barGap: 4,
    topPadding: 50,
    rightPadding: 75,
    leftPadding: 75,
    gridLineStartPadding: 35,
    fontSize: 11,
    sectionFontSize: 11,
    numberSectionStyles: 4,
    axisFormat: "%Y-%m-%d",
    topAxis: !1,
    displayMode: "",
    weekday: "sunday"
  },
  journey: {
    useMaxWidth: !0,
    diagramMarginX: 50,
    diagramMarginY: 10,
    leftMargin: 150,
    maxLabelWidth: 360,
    width: 150,
    height: 50,
    boxMargin: 10,
    boxTextMargin: 5,
    noteMargin: 10,
    messageMargin: 35,
    messageAlign: "center",
    bottomMarginAdj: 1,
    rightAngles: !1,
    taskFontSize: 14,
    taskFontFamily: '"Open Sans", sans-serif',
    taskMargin: 50,
    activationWidth: 10,
    textPlacement: "fo",
    actorColours: [
      "#8FBC8F",
      "#7CFC00",
      "#00FFFF",
      "#20B2AA",
      "#B0E0E6",
      "#FFFFE0"
    ],
    sectionFills: [
      "#191970",
      "#8B008B",
      "#4B0082",
      "#2F4F4F",
      "#800000",
      "#8B4513",
      "#00008B"
    ],
    sectionColours: [
      "#fff"
    ],
    titleColor: "",
    titleFontFamily: '"trebuchet ms", verdana, arial, sans-serif',
    titleFontSize: "4ex"
  },
  class: {
    useMaxWidth: !0,
    titleTopMargin: 25,
    arrowMarkerAbsolute: !1,
    dividerMargin: 10,
    padding: 5,
    textHeight: 10,
    defaultRenderer: "dagre-wrapper",
    htmlLabels: !1,
    hideEmptyMembersBox: !1
  },
  state: {
    useMaxWidth: !0,
    titleTopMargin: 25,
    dividerMargin: 10,
    sizeUnit: 5,
    padding: 8,
    textHeight: 10,
    titleShift: -15,
    noteMargin: 10,
    forkWidth: 70,
    forkHeight: 7,
    miniPadding: 2,
    fontSizeFactor: 5.02,
    fontSize: 24,
    labelHeight: 16,
    edgeLengthFactor: "20",
    compositTitleSize: 35,
    radius: 5,
    defaultRenderer: "dagre-wrapper"
  },
  er: {
    useMaxWidth: !0,
    titleTopMargin: 25,
    diagramPadding: 20,
    layoutDirection: "TB",
    minEntityWidth: 100,
    minEntityHeight: 75,
    entityPadding: 15,
    nodeSpacing: 140,
    rankSpacing: 80,
    stroke: "gray",
    fill: "honeydew",
    fontSize: 12
  },
  pie: {
    useMaxWidth: !0,
    textPosition: 0.75
  },
  quadrantChart: {
    useMaxWidth: !0,
    chartWidth: 500,
    chartHeight: 500,
    titleFontSize: 20,
    titlePadding: 10,
    quadrantPadding: 5,
    xAxisLabelPadding: 5,
    yAxisLabelPadding: 5,
    xAxisLabelFontSize: 16,
    yAxisLabelFontSize: 16,
    quadrantLabelFontSize: 16,
    quadrantTextTopPadding: 5,
    pointTextPadding: 5,
    pointLabelFontSize: 12,
    pointRadius: 5,
    xAxisPosition: "top",
    yAxisPosition: "left",
    quadrantInternalBorderStrokeWidth: 1,
    quadrantExternalBorderStrokeWidth: 2
  },
  xyChart: {
    useMaxWidth: !0,
    width: 700,
    height: 500,
    titleFontSize: 20,
    titlePadding: 10,
    showDataLabel: !1,
    showTitle: !0,
    xAxis: {
      $ref: "#/$defs/XYChartAxisConfig",
      showLabel: !0,
      labelFontSize: 14,
      labelPadding: 5,
      showTitle: !0,
      titleFontSize: 16,
      titlePadding: 5,
      showTick: !0,
      tickLength: 5,
      tickWidth: 2,
      showAxisLine: !0,
      axisLineWidth: 2
    },
    yAxis: {
      $ref: "#/$defs/XYChartAxisConfig",
      showLabel: !0,
      labelFontSize: 14,
      labelPadding: 5,
      showTitle: !0,
      titleFontSize: 16,
      titlePadding: 5,
      showTick: !0,
      tickLength: 5,
      tickWidth: 2,
      showAxisLine: !0,
      axisLineWidth: 2
    },
    chartOrientation: "vertical",
    plotReservedSpacePercent: 50
  },
  requirement: {
    useMaxWidth: !0,
    rect_fill: "#f9f9f9",
    text_color: "#333",
    rect_border_size: "0.5px",
    rect_border_color: "#bbb",
    rect_min_width: 200,
    rect_min_height: 200,
    fontSize: 14,
    rect_padding: 10,
    line_height: 20
  },
  mindmap: {
    useMaxWidth: !0,
    padding: 10,
    maxNodeWidth: 200
  },
  kanban: {
    useMaxWidth: !0,
    padding: 8,
    sectionWidth: 200,
    ticketBaseUrl: ""
  },
  timeline: {
    useMaxWidth: !0,
    diagramMarginX: 50,
    diagramMarginY: 10,
    leftMargin: 150,
    width: 150,
    height: 50,
    boxMargin: 10,
    boxTextMargin: 5,
    noteMargin: 10,
    messageMargin: 35,
    messageAlign: "center",
    bottomMarginAdj: 1,
    rightAngles: !1,
    taskFontSize: 14,
    taskFontFamily: '"Open Sans", sans-serif',
    taskMargin: 50,
    activationWidth: 10,
    textPlacement: "fo",
    actorColours: [
      "#8FBC8F",
      "#7CFC00",
      "#00FFFF",
      "#20B2AA",
      "#B0E0E6",
      "#FFFFE0"
    ],
    sectionFills: [
      "#191970",
      "#8B008B",
      "#4B0082",
      "#2F4F4F",
      "#800000",
      "#8B4513",
      "#00008B"
    ],
    sectionColours: [
      "#fff"
    ],
    disableMulticolor: !1
  },
  gitGraph: {
    useMaxWidth: !0,
    titleTopMargin: 25,
    diagramPadding: 8,
    nodeLabel: {
      width: 75,
      height: 100,
      x: -25,
      y: 0
    },
    mainBranchName: "main",
    mainBranchOrder: 0,
    showCommitLabel: !0,
    showBranches: !0,
    rotateCommitLabel: !0,
    parallelCommits: !1,
    arrowMarkerAbsolute: !1
  },
  c4: {
    useMaxWidth: !0,
    diagramMarginX: 50,
    diagramMarginY: 10,
    c4ShapeMargin: 50,
    c4ShapePadding: 20,
    width: 216,
    height: 60,
    boxMargin: 10,
    c4ShapeInRow: 4,
    nextLinePaddingX: 0,
    c4BoundaryInRow: 2,
    personFontSize: 14,
    personFontFamily: '"Open Sans", sans-serif',
    personFontWeight: "normal",
    external_personFontSize: 14,
    external_personFontFamily: '"Open Sans", sans-serif',
    external_personFontWeight: "normal",
    systemFontSize: 14,
    systemFontFamily: '"Open Sans", sans-serif',
    systemFontWeight: "normal",
    external_systemFontSize: 14,
    external_systemFontFamily: '"Open Sans", sans-serif',
    external_systemFontWeight: "normal",
    system_dbFontSize: 14,
    system_dbFontFamily: '"Open Sans", sans-serif',
    system_dbFontWeight: "normal",
    external_system_dbFontSize: 14,
    external_system_dbFontFamily: '"Open Sans", sans-serif',
    external_system_dbFontWeight: "normal",
    system_queueFontSize: 14,
    system_queueFontFamily: '"Open Sans", sans-serif',
    system_queueFontWeight: "normal",
    external_system_queueFontSize: 14,
    external_system_queueFontFamily: '"Open Sans", sans-serif',
    external_system_queueFontWeight: "normal",
    boundaryFontSize: 14,
    boundaryFontFamily: '"Open Sans", sans-serif',
    boundaryFontWeight: "normal",
    messageFontSize: 12,
    messageFontFamily: '"Open Sans", sans-serif',
    messageFontWeight: "normal",
    containerFontSize: 14,
    containerFontFamily: '"Open Sans", sans-serif',
    containerFontWeight: "normal",
    external_containerFontSize: 14,
    external_containerFontFamily: '"Open Sans", sans-serif',
    external_containerFontWeight: "normal",
    container_dbFontSize: 14,
    container_dbFontFamily: '"Open Sans", sans-serif',
    container_dbFontWeight: "normal",
    external_container_dbFontSize: 14,
    external_container_dbFontFamily: '"Open Sans", sans-serif',
    external_container_dbFontWeight: "normal",
    container_queueFontSize: 14,
    container_queueFontFamily: '"Open Sans", sans-serif',
    container_queueFontWeight: "normal",
    external_container_queueFontSize: 14,
    external_container_queueFontFamily: '"Open Sans", sans-serif',
    external_container_queueFontWeight: "normal",
    componentFontSize: 14,
    componentFontFamily: '"Open Sans", sans-serif',
    componentFontWeight: "normal",
    external_componentFontSize: 14,
    external_componentFontFamily: '"Open Sans", sans-serif',
    external_componentFontWeight: "normal",
    component_dbFontSize: 14,
    component_dbFontFamily: '"Open Sans", sans-serif',
    component_dbFontWeight: "normal",
    external_component_dbFontSize: 14,
    external_component_dbFontFamily: '"Open Sans", sans-serif',
    external_component_dbFontWeight: "normal",
    component_queueFontSize: 14,
    component_queueFontFamily: '"Open Sans", sans-serif',
    component_queueFontWeight: "normal",
    external_component_queueFontSize: 14,
    external_component_queueFontFamily: '"Open Sans", sans-serif',
    external_component_queueFontWeight: "normal",
    wrap: !0,
    wrapPadding: 10,
    person_bg_color: "#08427B",
    person_border_color: "#073B6F",
    external_person_bg_color: "#686868",
    external_person_border_color: "#8A8A8A",
    system_bg_color: "#1168BD",
    system_border_color: "#3C7FC0",
    system_db_bg_color: "#1168BD",
    system_db_border_color: "#3C7FC0",
    system_queue_bg_color: "#1168BD",
    system_queue_border_color: "#3C7FC0",
    external_system_bg_color: "#999999",
    external_system_border_color: "#8A8A8A",
    external_system_db_bg_color: "#999999",
    external_system_db_border_color: "#8A8A8A",
    external_system_queue_bg_color: "#999999",
    external_system_queue_border_color: "#8A8A8A",
    container_bg_color: "#438DD5",
    container_border_color: "#3C7FC0",
    container_db_bg_color: "#438DD5",
    container_db_border_color: "#3C7FC0",
    container_queue_bg_color: "#438DD5",
    container_queue_border_color: "#3C7FC0",
    external_container_bg_color: "#B3B3B3",
    external_container_border_color: "#A6A6A6",
    external_container_db_bg_color: "#B3B3B3",
    external_container_db_border_color: "#A6A6A6",
    external_container_queue_bg_color: "#B3B3B3",
    external_container_queue_border_color: "#A6A6A6",
    component_bg_color: "#85BBF0",
    component_border_color: "#78A8D8",
    component_db_bg_color: "#85BBF0",
    component_db_border_color: "#78A8D8",
    component_queue_bg_color: "#85BBF0",
    component_queue_border_color: "#78A8D8",
    external_component_bg_color: "#CCCCCC",
    external_component_border_color: "#BFBFBF",
    external_component_db_bg_color: "#CCCCCC",
    external_component_db_border_color: "#BFBFBF",
    external_component_queue_bg_color: "#CCCCCC",
    external_component_queue_border_color: "#BFBFBF"
  },
  sankey: {
    useMaxWidth: !0,
    width: 600,
    height: 400,
    linkColor: "gradient",
    nodeAlignment: "justify",
    showValues: !0,
    prefix: "",
    suffix: ""
  },
  block: {
    useMaxWidth: !0,
    padding: 8
  },
  packet: {
    useMaxWidth: !0,
    rowHeight: 32,
    bitWidth: 32,
    bitsPerRow: 32,
    showBits: !0,
    paddingX: 5,
    paddingY: 5
  },
  architecture: {
    useMaxWidth: !0,
    padding: 40,
    iconSize: 80,
    fontSize: 16
  },
  radar: {
    useMaxWidth: !0,
    width: 600,
    height: 600,
    marginTop: 50,
    marginRight: 50,
    marginBottom: 50,
    marginLeft: 50,
    axisScaleFactor: 1,
    axisLabelFactor: 1.05,
    curveTension: 0.17
  },
  theme: "default",
  look: "classic",
  handDrawnSeed: 0,
  layout: "dagre",
  maxTextSize: 5e4,
  maxEdges: 500,
  darkMode: !1,
  fontFamily: '"trebuchet ms", verdana, arial, sans-serif;',
  logLevel: 5,
  securityLevel: "strict",
  startOnLoad: !0,
  arrowMarkerAbsolute: !1,
  secure: [
    "secure",
    "securityLevel",
    "startOnLoad",
    "maxTextSize",
    "suppressErrorRendering",
    "maxEdges"
  ],
  legacyMathML: !1,
  forceLegacyMathML: !1,
  deterministicIds: !1,
  fontSize: 16,
  markdownAutoWrap: !0,
  suppressErrorRendering: !1
}, Hl = {
  ...Vt,
  // Set, even though they're `undefined` so that `configKeys` finds these keys
  // TODO: Should we replace these with `null` so that they can go in the JSON Schema?
  deterministicIDSeed: void 0,
  elk: {
    // mergeEdges is needed here to be considered
    mergeEdges: !1,
    nodePlacementStrategy: "BRANDES_KOEPF"
  },
  themeCSS: void 0,
  // add non-JSON default config values
  themeVariables: ue.default.getThemeVariables(),
  sequence: {
    ...Vt.sequence,
    messageFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.messageFontFamily,
        fontSize: this.messageFontSize,
        fontWeight: this.messageFontWeight
      };
    }, "messageFont"),
    noteFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.noteFontFamily,
        fontSize: this.noteFontSize,
        fontWeight: this.noteFontWeight
      };
    }, "noteFont"),
    actorFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.actorFontFamily,
        fontSize: this.actorFontSize,
        fontWeight: this.actorFontWeight
      };
    }, "actorFont")
  },
  class: {
    hideEmptyMembersBox: !1
  },
  gantt: {
    ...Vt.gantt,
    tickInterval: void 0,
    useWidth: void 0
    // can probably be removed since `configKeys` already includes this
  },
  c4: {
    ...Vt.c4,
    useWidth: void 0,
    personFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.personFontFamily,
        fontSize: this.personFontSize,
        fontWeight: this.personFontWeight
      };
    }, "personFont"),
    flowchart: {
      ...Vt.flowchart,
      inheritDir: !1
      // default to legacy behavior
    },
    external_personFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_personFontFamily,
        fontSize: this.external_personFontSize,
        fontWeight: this.external_personFontWeight
      };
    }, "external_personFont"),
    systemFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.systemFontFamily,
        fontSize: this.systemFontSize,
        fontWeight: this.systemFontWeight
      };
    }, "systemFont"),
    external_systemFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_systemFontFamily,
        fontSize: this.external_systemFontSize,
        fontWeight: this.external_systemFontWeight
      };
    }, "external_systemFont"),
    system_dbFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.system_dbFontFamily,
        fontSize: this.system_dbFontSize,
        fontWeight: this.system_dbFontWeight
      };
    }, "system_dbFont"),
    external_system_dbFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_system_dbFontFamily,
        fontSize: this.external_system_dbFontSize,
        fontWeight: this.external_system_dbFontWeight
      };
    }, "external_system_dbFont"),
    system_queueFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.system_queueFontFamily,
        fontSize: this.system_queueFontSize,
        fontWeight: this.system_queueFontWeight
      };
    }, "system_queueFont"),
    external_system_queueFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_system_queueFontFamily,
        fontSize: this.external_system_queueFontSize,
        fontWeight: this.external_system_queueFontWeight
      };
    }, "external_system_queueFont"),
    containerFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.containerFontFamily,
        fontSize: this.containerFontSize,
        fontWeight: this.containerFontWeight
      };
    }, "containerFont"),
    external_containerFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_containerFontFamily,
        fontSize: this.external_containerFontSize,
        fontWeight: this.external_containerFontWeight
      };
    }, "external_containerFont"),
    container_dbFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.container_dbFontFamily,
        fontSize: this.container_dbFontSize,
        fontWeight: this.container_dbFontWeight
      };
    }, "container_dbFont"),
    external_container_dbFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_container_dbFontFamily,
        fontSize: this.external_container_dbFontSize,
        fontWeight: this.external_container_dbFontWeight
      };
    }, "external_container_dbFont"),
    container_queueFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.container_queueFontFamily,
        fontSize: this.container_queueFontSize,
        fontWeight: this.container_queueFontWeight
      };
    }, "container_queueFont"),
    external_container_queueFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_container_queueFontFamily,
        fontSize: this.external_container_queueFontSize,
        fontWeight: this.external_container_queueFontWeight
      };
    }, "external_container_queueFont"),
    componentFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.componentFontFamily,
        fontSize: this.componentFontSize,
        fontWeight: this.componentFontWeight
      };
    }, "componentFont"),
    external_componentFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_componentFontFamily,
        fontSize: this.external_componentFontSize,
        fontWeight: this.external_componentFontWeight
      };
    }, "external_componentFont"),
    component_dbFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.component_dbFontFamily,
        fontSize: this.component_dbFontSize,
        fontWeight: this.component_dbFontWeight
      };
    }, "component_dbFont"),
    external_component_dbFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_component_dbFontFamily,
        fontSize: this.external_component_dbFontSize,
        fontWeight: this.external_component_dbFontWeight
      };
    }, "external_component_dbFont"),
    component_queueFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.component_queueFontFamily,
        fontSize: this.component_queueFontSize,
        fontWeight: this.component_queueFontWeight
      };
    }, "component_queueFont"),
    external_component_queueFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.external_component_queueFontFamily,
        fontSize: this.external_component_queueFontSize,
        fontWeight: this.external_component_queueFontWeight
      };
    }, "external_component_queueFont"),
    boundaryFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.boundaryFontFamily,
        fontSize: this.boundaryFontSize,
        fontWeight: this.boundaryFontWeight
      };
    }, "boundaryFont"),
    messageFont: /* @__PURE__ */ p(function() {
      return {
        fontFamily: this.messageFontFamily,
        fontSize: this.messageFontSize,
        fontWeight: this.messageFontWeight
      };
    }, "messageFont")
  },
  pie: {
    ...Vt.pie,
    useWidth: 984
  },
  xyChart: {
    ...Vt.xyChart,
    useWidth: void 0
  },
  requirement: {
    ...Vt.requirement,
    useWidth: void 0
  },
  packet: {
    ...Vt.packet
  },
  radar: {
    ...Vt.radar
  },
  treemap: {
    useMaxWidth: !0,
    padding: 10,
    diagramPadding: 8,
    showValues: !0,
    nodeWidth: 100,
    nodeHeight: 40,
    borderWidth: 1,
    valueFontSize: 12,
    labelFontSize: 14,
    valueFormat: ","
  }
}, jl = /* @__PURE__ */ p((e, t = "") => Object.keys(e).reduce((r, i) => Array.isArray(e[i]) ? r : typeof e[i] == "object" && e[i] !== null ? [...r, t + i, ...jl(e[i], "")] : [...r, t + i], []), "keyify"), kg = new Set(jl(Hl, "")), Yl = Hl, Hi = /* @__PURE__ */ p((e) => {
  if (F.debug("sanitizeDirective called with", e), !(typeof e != "object" || e == null)) {
    if (Array.isArray(e)) {
      e.forEach((t) => Hi(t));
      return;
    }
    for (const t of Object.keys(e)) {
      if (F.debug("Checking key", t), t.startsWith("__") || t.includes("proto") || t.includes("constr") || !kg.has(t) || e[t] == null) {
        F.debug("sanitize deleting key: ", t), delete e[t];
        continue;
      }
      if (typeof e[t] == "object") {
        F.debug("sanitizing object", t), Hi(e[t]);
        continue;
      }
      const r = ["themeCSS", "fontFamily", "altFontFamily"];
      for (const i of r)
        t.includes(i) && (F.debug("sanitizing css option", t), e[t] = wg(e[t]));
    }
    if (e.themeVariables)
      for (const t of Object.keys(e.themeVariables)) {
        const r = e.themeVariables[t];
        r != null && r.match && !r.match(/^[\d "#%(),.;A-Za-z]+$/) && (e.themeVariables[t] = "");
      }
    F.debug("After sanitization", e);
  }
}, "sanitizeDirective"), wg = /* @__PURE__ */ p((e) => {
  let t = 0, r = 0;
  for (const i of e) {
    if (t < r)
      return "{ /* ERROR: Unbalanced CSS */ }";
    i === "{" ? t++ : i === "}" && r++;
  }
  return t !== r ? "{ /* ERROR: Unbalanced CSS */ }" : e;
}, "sanitizeCss"), gr = Object.freeze(Yl), Dt = vt({}, gr), Gl, mr = [], Xr = vt({}, gr), Ba = /* @__PURE__ */ p((e, t) => {
  let r = vt({}, e), i = {};
  for (const a of t)
    Vl(a), i = vt(i, a);
  if (r = vt(r, i), i.theme && i.theme in ue) {
    const a = vt({}, Gl), n = vt(
      a.themeVariables || {},
      i.themeVariables
    );
    r.theme && r.theme in ue && (r.themeVariables = ue[r.theme].getThemeVariables(n));
  }
  return Xr = r, Zl(Xr), Xr;
}, "updateCurrentConfig"), _g = /* @__PURE__ */ p((e) => (Dt = vt({}, gr), Dt = vt(Dt, e), e.theme && ue[e.theme] && (Dt.themeVariables = ue[e.theme].getThemeVariables(e.themeVariables)), Ba(Dt, mr), Dt), "setSiteConfig"), vg = /* @__PURE__ */ p((e) => {
  Gl = vt({}, e);
}, "saveConfigFromInitialize"), Sg = /* @__PURE__ */ p((e) => (Dt = vt(Dt, e), Ba(Dt, mr), Dt), "updateSiteConfig"), Ul = /* @__PURE__ */ p(() => vt({}, Dt), "getSiteConfig"), Xl = /* @__PURE__ */ p((e) => (Zl(e), vt(Xr, e), It()), "setConfig"), It = /* @__PURE__ */ p(() => vt({}, Xr), "getConfig"), Vl = /* @__PURE__ */ p((e) => {
  e && (["secure", ...Dt.secure ?? []].forEach((t) => {
    Object.hasOwn(e, t) && (F.debug(`Denied attempt to modify a secure key ${t}`, e[t]), delete e[t]);
  }), Object.keys(e).forEach((t) => {
    t.startsWith("__") && delete e[t];
  }), Object.keys(e).forEach((t) => {
    typeof e[t] == "string" && (e[t].includes("<") || e[t].includes(">") || e[t].includes("url(data:")) && delete e[t], typeof e[t] == "object" && Vl(e[t]);
  }));
}, "sanitize"), Tg = /* @__PURE__ */ p((e) => {
  var t;
  Hi(e), e.fontFamily && !((t = e.themeVariables) != null && t.fontFamily) && (e.themeVariables = {
    ...e.themeVariables,
    fontFamily: e.fontFamily
  }), mr.push(e), Ba(Dt, mr);
}, "addDirective"), ji = /* @__PURE__ */ p((e = Dt) => {
  mr = [], Ba(e, mr);
}, "reset"), Bg = {
  LAZY_LOAD_DEPRECATED: "The configuration options lazyLoadedDiagrams and loadExternalDiagramsAtStartup are deprecated. Please use registerExternalDiagrams instead."
}, fo = {}, Lg = /* @__PURE__ */ p((e) => {
  fo[e] || (F.warn(Bg[e]), fo[e] = !0);
}, "issueWarning"), Zl = /* @__PURE__ */ p((e) => {
  e && (e.lazyLoadedDiagrams || e.loadExternalDiagramsAtStartup) && Lg("LAZY_LOAD_DEPRECATED");
}, "checkConfig"), hi = /<br\s*\/?>/gi, Mg = /* @__PURE__ */ p((e) => e ? Jl(e).replace(/\\n/g, "#br#").split("#br#") : [""], "getRows"), $g = /* @__PURE__ */ (() => {
  let e = !1;
  return () => {
    e || (Kl(), e = !0);
  };
})();
function Kl() {
  const e = "data-temp-href-target";
  pr.addHook("beforeSanitizeAttributes", (t) => {
    t instanceof Element && t.tagName === "A" && t.hasAttribute("target") && t.setAttribute(e, t.getAttribute("target") ?? "");
  }), pr.addHook("afterSanitizeAttributes", (t) => {
    t instanceof Element && t.tagName === "A" && t.hasAttribute(e) && (t.setAttribute("target", t.getAttribute(e) ?? ""), t.removeAttribute(e), t.getAttribute("target") === "_blank" && t.setAttribute("rel", "noopener"));
  });
}
p(Kl, "setupDompurifyHooks");
var Ql = /* @__PURE__ */ p((e) => ($g(), pr.sanitize(e)), "removeScript"), po = /* @__PURE__ */ p((e, t) => {
  var r;
  if (((r = t.flowchart) == null ? void 0 : r.htmlLabels) !== !1) {
    const i = t.securityLevel;
    i === "antiscript" || i === "strict" ? e = Ql(e) : i !== "loose" && (e = Jl(e), e = e.replace(/</g, "&lt;").replace(/>/g, "&gt;"), e = e.replace(/=/g, "&equals;"), e = Og(e));
  }
  return e;
}, "sanitizeMore"), qe = /* @__PURE__ */ p((e, t) => e && (t.dompurifyConfig ? e = pr.sanitize(po(e, t), t.dompurifyConfig).toString() : e = pr.sanitize(po(e, t), {
  FORBID_TAGS: ["style"]
}).toString(), e), "sanitizeText"), Ag = /* @__PURE__ */ p((e, t) => typeof e == "string" ? qe(e, t) : e.flat().map((r) => qe(r, t)), "sanitizeTextOrArray"), Fg = /* @__PURE__ */ p((e) => hi.test(e), "hasBreaks"), Eg = /* @__PURE__ */ p((e) => e.split(hi), "splitBreaks"), Og = /* @__PURE__ */ p((e) => e.replace(/#br#/g, "<br/>"), "placeholderToBreak"), Jl = /* @__PURE__ */ p((e) => e.replace(hi, "#br#"), "breakToPlaceholder"), tc = /* @__PURE__ */ p((e) => {
  let t = "";
  return e && (t = window.location.protocol + "//" + window.location.host + window.location.pathname + window.location.search, t = CSS.escape(t)), t;
}, "getUrl"), bt = /* @__PURE__ */ p((e) => !(e === !1 || ["false", "null", "0"].includes(String(e).trim().toLowerCase())), "evaluate"), Dg = /* @__PURE__ */ p(function(...e) {
  const t = e.filter((r) => !isNaN(r));
  return Math.max(...t);
}, "getMax"), Rg = /* @__PURE__ */ p(function(...e) {
  const t = e.filter((r) => !isNaN(r));
  return Math.min(...t);
}, "getMin"), go = /* @__PURE__ */ p(function(e) {
  const t = e.split(/(,)/), r = [];
  for (let i = 0; i < t.length; i++) {
    let a = t[i];
    if (a === "," && i > 0 && i + 1 < t.length) {
      const n = t[i - 1], o = t[i + 1];
      Pg(n, o) && (a = n + "," + o, i++, r.pop());
    }
    r.push(Ig(a));
  }
  return r.join("");
}, "parseGenericTypes"), yn = /* @__PURE__ */ p((e, t) => Math.max(0, e.split(t).length - 1), "countOccurrence"), Pg = /* @__PURE__ */ p((e, t) => {
  const r = yn(e, "~"), i = yn(t, "~");
  return r === 1 && i === 1;
}, "shouldCombineSets"), Ig = /* @__PURE__ */ p((e) => {
  const t = yn(e, "~");
  let r = !1;
  if (t <= 1)
    return e;
  t % 2 !== 0 && e.startsWith("~") && (e = e.substring(1), r = !0);
  const i = [...e];
  let a = i.indexOf("~"), n = i.lastIndexOf("~");
  for (; a !== -1 && n !== -1 && a !== n; )
    i[a] = "<", i[n] = ">", a = i.indexOf("~"), n = i.lastIndexOf("~");
  return r && i.unshift("~"), i.join("");
}, "processSet"), mo = /* @__PURE__ */ p(() => window.MathMLElement !== void 0, "isMathMLSupported"), xn = /\$\$(.*)\$\$/g, yr = /* @__PURE__ */ p((e) => {
  var t;
  return (((t = e.match(xn)) == null ? void 0 : t.length) ?? 0) > 0;
}, "hasKatex"), _T = /* @__PURE__ */ p(async (e, t) => {
  e = await fs(e, t);
  const r = document.createElement("div");
  r.innerHTML = e, r.id = "katex-temp", r.style.visibility = "hidden", r.style.position = "absolute", r.style.top = "0";
  const i = document.querySelector("body");
  i == null || i.insertAdjacentElement("beforeend", r);
  const a = { width: r.clientWidth, height: r.clientHeight };
  return r.remove(), a;
}, "calculateMathMLDimensions"), fs = /* @__PURE__ */ p(async (e, t) => {
  if (!yr(e))
    return e;
  if (!(mo() || t.legacyMathML || t.forceLegacyMathML))
    return e.replace(xn, "MathML is unsupported in this environment.");
  {
    const { default: r } = await import("./Index-BG2POTv1.js").then((a) => a.k), i = t.forceLegacyMathML || !mo() && t.legacyMathML ? "htmlAndMathml" : "mathml";
    return e.split(hi).map(
      (a) => yr(a) ? `<div style="display: flex; align-items: center; justify-content: center; white-space: nowrap;">${a}</div>` : `<div>${a}</div>`
    ).join("").replace(
      xn,
      (a, n) => r.renderToString(n, {
        throwOnError: !0,
        displayMode: !0,
        output: i
      }).replace(/\n/g, " ").replace(/<annotation.*<\/annotation>/g, "")
    );
  }
}, "renderKatex"), vr = {
  getRows: Mg,
  sanitizeText: qe,
  sanitizeTextOrArray: Ag,
  hasBreaks: Fg,
  splitBreaks: Eg,
  lineBreakRegex: hi,
  removeScript: Ql,
  getUrl: tc,
  evaluate: bt,
  getMax: Dg,
  getMin: Rg
}, Ng = /* @__PURE__ */ p(function(e, t) {
  for (let r of t)
    e.attr(r[0], r[1]);
}, "d3Attrs"), zg = /* @__PURE__ */ p(function(e, t, r) {
  let i = /* @__PURE__ */ new Map();
  return r ? (i.set("width", "100%"), i.set("style", `max-width: ${t}px;`)) : (i.set("height", e), i.set("width", t)), i;
}, "calculateSvgSizeAttrs"), ec = /* @__PURE__ */ p(function(e, t, r, i) {
  const a = zg(t, r, i);
  Ng(e, a);
}, "configureSvgSize"), qg = /* @__PURE__ */ p(function(e, t, r, i) {
  const a = t.node().getBBox(), n = a.width, o = a.height;
  F.info(`SVG bounds: ${n}x${o}`, a);
  let s = 0, l = 0;
  F.info(`Graph bounds: ${s}x${l}`, e), s = n + r * 2, l = o + r * 2, F.info(`Calculated bounds: ${s}x${l}`), ec(t, l, s, i);
  const c = `${a.x - r} ${a.y - r} ${a.width + 2 * r} ${a.height + 2 * r}`;
  t.attr("viewBox", c);
}, "setupGraphViewbox"), $i = {}, Wg = /* @__PURE__ */ p((e, t, r) => {
  let i = "";
  return e in $i && $i[e] ? i = $i[e](r) : F.warn(`No theme found for ${e}`), ` & {
    font-family: ${r.fontFamily};
    font-size: ${r.fontSize};
    fill: ${r.textColor}
  }
  @keyframes edge-animation-frame {
    from {
      stroke-dashoffset: 0;
    }
  }
  @keyframes dash {
    to {
      stroke-dashoffset: 0;
    }
  }
  & .edge-animation-slow {
    stroke-dasharray: 9,5 !important;
    stroke-dashoffset: 900;
    animation: dash 50s linear infinite;
    stroke-linecap: round;
  }
  & .edge-animation-fast {
    stroke-dasharray: 9,5 !important;
    stroke-dashoffset: 900;
    animation: dash 20s linear infinite;
    stroke-linecap: round;
  }
  /* Classes common for multiple diagrams */

  & .error-icon {
    fill: ${r.errorBkgColor};
  }
  & .error-text {
    fill: ${r.errorTextColor};
    stroke: ${r.errorTextColor};
  }

  & .edge-thickness-normal {
    stroke-width: 1px;
  }
  & .edge-thickness-thick {
    stroke-width: 3.5px
  }
  & .edge-pattern-solid {
    stroke-dasharray: 0;
  }
  & .edge-thickness-invisible {
    stroke-width: 0;
    fill: none;
  }
  & .edge-pattern-dashed{
    stroke-dasharray: 3;
  }
  .edge-pattern-dotted {
    stroke-dasharray: 2;
  }

  & .marker {
    fill: ${r.lineColor};
    stroke: ${r.lineColor};
  }
  & .marker.cross {
    stroke: ${r.lineColor};
  }

  & svg {
    font-family: ${r.fontFamily};
    font-size: ${r.fontSize};
  }
   & p {
    margin: 0
   }

  ${i}

  ${t}
`;
}, "getStyles"), Hg = /* @__PURE__ */ p((e, t) => {
  t !== void 0 && ($i[e] = t);
}, "addStylesForDiagram"), jg = Wg, rc = {};
lg(rc, {
  clear: () => Yg,
  getAccDescription: () => Vg,
  getAccTitle: () => Ug,
  getDiagramTitle: () => Kg,
  setAccDescription: () => Xg,
  setAccTitle: () => Gg,
  setDiagramTitle: () => Zg
});
var ds = "", ps = "", gs = "", ms = /* @__PURE__ */ p((e) => qe(e, It()), "sanitizeText"), Yg = /* @__PURE__ */ p(() => {
  ds = "", gs = "", ps = "";
}, "clear"), Gg = /* @__PURE__ */ p((e) => {
  ds = ms(e).replace(/^\s+/g, "");
}, "setAccTitle"), Ug = /* @__PURE__ */ p(() => ds, "getAccTitle"), Xg = /* @__PURE__ */ p((e) => {
  gs = ms(e).replace(/\n\s+/g, `
`);
}, "setAccDescription"), Vg = /* @__PURE__ */ p(() => gs, "getAccDescription"), Zg = /* @__PURE__ */ p((e) => {
  ps = ms(e);
}, "setDiagramTitle"), Kg = /* @__PURE__ */ p(() => ps, "getDiagramTitle"), yo = F, Qg = hs, at = It, vT = Xl, ST = gr, La = /* @__PURE__ */ p((e) => qe(e, at()), "sanitizeText"), Jg = qg, t0 = /* @__PURE__ */ p(() => rc, "getCommonDb"), Yi = {}, Gi = /* @__PURE__ */ p((e, t, r) => {
  var i;
  Yi[e] && yo.warn(`Diagram with id ${e} already registered. Overwriting.`), Yi[e] = t, r && Wl(e, r), Hg(e, t.styles), (i = t.injectUtils) == null || i.call(
    t,
    yo,
    Qg,
    at,
    La,
    Jg,
    t0(),
    () => {
    }
  );
}, "registerDiagram"), bn = /* @__PURE__ */ p((e) => {
  if (e in Yi)
    return Yi[e];
  throw new e0(e);
}, "getDiagram"), fr, e0 = (fr = class extends Error {
  constructor(t) {
    super(`Diagram ${t} not found.`);
  }
}, p(fr, "DiagramNotFoundError"), fr);
function ys(e) {
  return typeof e > "u" || e === null;
}
p(ys, "isNothing");
function ic(e) {
  return typeof e == "object" && e !== null;
}
p(ic, "isObject");
function ac(e) {
  return Array.isArray(e) ? e : ys(e) ? [] : [e];
}
p(ac, "toArray");
function nc(e, t) {
  var r, i, a, n;
  if (t)
    for (n = Object.keys(t), r = 0, i = n.length; r < i; r += 1)
      a = n[r], e[a] = t[a];
  return e;
}
p(nc, "extend");
function sc(e, t) {
  var r = "", i;
  for (i = 0; i < t; i += 1)
    r += e;
  return r;
}
p(sc, "repeat");
function oc(e) {
  return e === 0 && Number.NEGATIVE_INFINITY === 1 / e;
}
p(oc, "isNegativeZero");
var r0 = ys, i0 = ic, a0 = ac, n0 = sc, s0 = oc, o0 = nc, xt = {
  isNothing: r0,
  isObject: i0,
  toArray: a0,
  repeat: n0,
  isNegativeZero: s0,
  extend: o0
};
function xs(e, t) {
  var r = "", i = e.reason || "(unknown reason)";
  return e.mark ? (e.mark.name && (r += 'in "' + e.mark.name + '" '), r += "(" + (e.mark.line + 1) + ":" + (e.mark.column + 1) + ")", !t && e.mark.snippet && (r += `

` + e.mark.snippet), i + " " + r) : i;
}
p(xs, "formatError");
function xr(e, t) {
  Error.call(this), this.name = "YAMLException", this.reason = e, this.mark = t, this.message = xs(this, !1), Error.captureStackTrace ? Error.captureStackTrace(this, this.constructor) : this.stack = new Error().stack || "";
}
p(xr, "YAMLException$1");
xr.prototype = Object.create(Error.prototype);
xr.prototype.constructor = xr;
xr.prototype.toString = /* @__PURE__ */ p(function(t) {
  return this.name + ": " + xs(this, t);
}, "toString");
var Rt = xr;
function Ai(e, t, r, i, a) {
  var n = "", o = "", s = Math.floor(a / 2) - 1;
  return i - t > s && (n = " ... ", t = i - s + n.length), r - i > s && (o = " ...", r = i + s - o.length), {
    str: n + e.slice(t, r).replace(/\t/g, "→") + o,
    pos: i - t + n.length
    // relative position
  };
}
p(Ai, "getLine");
function Fi(e, t) {
  return xt.repeat(" ", t - e.length) + e;
}
p(Fi, "padStart");
function lc(e, t) {
  if (t = Object.create(t || null), !e.buffer) return null;
  t.maxLength || (t.maxLength = 79), typeof t.indent != "number" && (t.indent = 1), typeof t.linesBefore != "number" && (t.linesBefore = 3), typeof t.linesAfter != "number" && (t.linesAfter = 2);
  for (var r = /\r?\n|\r|\0/g, i = [0], a = [], n, o = -1; n = r.exec(e.buffer); )
    a.push(n.index), i.push(n.index + n[0].length), e.position <= n.index && o < 0 && (o = i.length - 2);
  o < 0 && (o = i.length - 1);
  var s = "", l, c, h = Math.min(e.line + t.linesAfter, a.length).toString().length, u = t.maxLength - (t.indent + h + 3);
  for (l = 1; l <= t.linesBefore && !(o - l < 0); l++)
    c = Ai(
      e.buffer,
      i[o - l],
      a[o - l],
      e.position - (i[o] - i[o - l]),
      u
    ), s = xt.repeat(" ", t.indent) + Fi((e.line - l + 1).toString(), h) + " | " + c.str + `
` + s;
  for (c = Ai(e.buffer, i[o], a[o], e.position, u), s += xt.repeat(" ", t.indent) + Fi((e.line + 1).toString(), h) + " | " + c.str + `
`, s += xt.repeat("-", t.indent + h + 3 + c.pos) + `^
`, l = 1; l <= t.linesAfter && !(o + l >= a.length); l++)
    c = Ai(
      e.buffer,
      i[o + l],
      a[o + l],
      e.position - (i[o] - i[o + l]),
      u
    ), s += xt.repeat(" ", t.indent) + Fi((e.line + l + 1).toString(), h) + " | " + c.str + `
`;
  return s.replace(/\n$/, "");
}
p(lc, "makeSnippet");
var l0 = lc, c0 = [
  "kind",
  "multi",
  "resolve",
  "construct",
  "instanceOf",
  "predicate",
  "represent",
  "representName",
  "defaultStyle",
  "styleAliases"
], h0 = [
  "scalar",
  "sequence",
  "mapping"
];
function cc(e) {
  var t = {};
  return e !== null && Object.keys(e).forEach(function(r) {
    e[r].forEach(function(i) {
      t[String(i)] = r;
    });
  }), t;
}
p(cc, "compileStyleAliases");
function hc(e, t) {
  if (t = t || {}, Object.keys(t).forEach(function(r) {
    if (c0.indexOf(r) === -1)
      throw new Rt('Unknown option "' + r + '" is met in definition of "' + e + '" YAML type.');
  }), this.options = t, this.tag = e, this.kind = t.kind || null, this.resolve = t.resolve || function() {
    return !0;
  }, this.construct = t.construct || function(r) {
    return r;
  }, this.instanceOf = t.instanceOf || null, this.predicate = t.predicate || null, this.represent = t.represent || null, this.representName = t.representName || null, this.defaultStyle = t.defaultStyle || null, this.multi = t.multi || !1, this.styleAliases = cc(t.styleAliases || null), h0.indexOf(this.kind) === -1)
    throw new Rt('Unknown kind "' + this.kind + '" is specified for "' + e + '" YAML type.');
}
p(hc, "Type$1");
var Bt = hc;
function Cn(e, t) {
  var r = [];
  return e[t].forEach(function(i) {
    var a = r.length;
    r.forEach(function(n, o) {
      n.tag === i.tag && n.kind === i.kind && n.multi === i.multi && (a = o);
    }), r[a] = i;
  }), r;
}
p(Cn, "compileList");
function uc() {
  var e = {
    scalar: {},
    sequence: {},
    mapping: {},
    fallback: {},
    multi: {
      scalar: [],
      sequence: [],
      mapping: [],
      fallback: []
    }
  }, t, r;
  function i(a) {
    a.multi ? (e.multi[a.kind].push(a), e.multi.fallback.push(a)) : e[a.kind][a.tag] = e.fallback[a.tag] = a;
  }
  for (p(i, "collectType"), t = 0, r = arguments.length; t < r; t += 1)
    arguments[t].forEach(i);
  return e;
}
p(uc, "compileMap");
function Ui(e) {
  return this.extend(e);
}
p(Ui, "Schema$1");
Ui.prototype.extend = /* @__PURE__ */ p(function(t) {
  var r = [], i = [];
  if (t instanceof Bt)
    i.push(t);
  else if (Array.isArray(t))
    i = i.concat(t);
  else if (t && (Array.isArray(t.implicit) || Array.isArray(t.explicit)))
    t.implicit && (r = r.concat(t.implicit)), t.explicit && (i = i.concat(t.explicit));
  else
    throw new Rt("Schema.extend argument should be a Type, [ Type ], or a schema definition ({ implicit: [...], explicit: [...] })");
  r.forEach(function(n) {
    if (!(n instanceof Bt))
      throw new Rt("Specified list of YAML types (or a single Type object) contains a non-Type object.");
    if (n.loadKind && n.loadKind !== "scalar")
      throw new Rt("There is a non-scalar type in the implicit list of a schema. Implicit resolving of such types is not supported.");
    if (n.multi)
      throw new Rt("There is a multi type in the implicit list of a schema. Multi tags can only be listed as explicit.");
  }), i.forEach(function(n) {
    if (!(n instanceof Bt))
      throw new Rt("Specified list of YAML types (or a single Type object) contains a non-Type object.");
  });
  var a = Object.create(Ui.prototype);
  return a.implicit = (this.implicit || []).concat(r), a.explicit = (this.explicit || []).concat(i), a.compiledImplicit = Cn(a, "implicit"), a.compiledExplicit = Cn(a, "explicit"), a.compiledTypeMap = uc(a.compiledImplicit, a.compiledExplicit), a;
}, "extend");
var u0 = Ui, f0 = new Bt("tag:yaml.org,2002:str", {
  kind: "scalar",
  construct: /* @__PURE__ */ p(function(e) {
    return e !== null ? e : "";
  }, "construct")
}), d0 = new Bt("tag:yaml.org,2002:seq", {
  kind: "sequence",
  construct: /* @__PURE__ */ p(function(e) {
    return e !== null ? e : [];
  }, "construct")
}), p0 = new Bt("tag:yaml.org,2002:map", {
  kind: "mapping",
  construct: /* @__PURE__ */ p(function(e) {
    return e !== null ? e : {};
  }, "construct")
}), g0 = new u0({
  explicit: [
    f0,
    d0,
    p0
  ]
});
function fc(e) {
  if (e === null) return !0;
  var t = e.length;
  return t === 1 && e === "~" || t === 4 && (e === "null" || e === "Null" || e === "NULL");
}
p(fc, "resolveYamlNull");
function dc() {
  return null;
}
p(dc, "constructYamlNull");
function pc(e) {
  return e === null;
}
p(pc, "isNull");
var m0 = new Bt("tag:yaml.org,2002:null", {
  kind: "scalar",
  resolve: fc,
  construct: dc,
  predicate: pc,
  represent: {
    canonical: /* @__PURE__ */ p(function() {
      return "~";
    }, "canonical"),
    lowercase: /* @__PURE__ */ p(function() {
      return "null";
    }, "lowercase"),
    uppercase: /* @__PURE__ */ p(function() {
      return "NULL";
    }, "uppercase"),
    camelcase: /* @__PURE__ */ p(function() {
      return "Null";
    }, "camelcase"),
    empty: /* @__PURE__ */ p(function() {
      return "";
    }, "empty")
  },
  defaultStyle: "lowercase"
});
function gc(e) {
  if (e === null) return !1;
  var t = e.length;
  return t === 4 && (e === "true" || e === "True" || e === "TRUE") || t === 5 && (e === "false" || e === "False" || e === "FALSE");
}
p(gc, "resolveYamlBoolean");
function mc(e) {
  return e === "true" || e === "True" || e === "TRUE";
}
p(mc, "constructYamlBoolean");
function yc(e) {
  return Object.prototype.toString.call(e) === "[object Boolean]";
}
p(yc, "isBoolean");
var y0 = new Bt("tag:yaml.org,2002:bool", {
  kind: "scalar",
  resolve: gc,
  construct: mc,
  predicate: yc,
  represent: {
    lowercase: /* @__PURE__ */ p(function(e) {
      return e ? "true" : "false";
    }, "lowercase"),
    uppercase: /* @__PURE__ */ p(function(e) {
      return e ? "TRUE" : "FALSE";
    }, "uppercase"),
    camelcase: /* @__PURE__ */ p(function(e) {
      return e ? "True" : "False";
    }, "camelcase")
  },
  defaultStyle: "lowercase"
});
function xc(e) {
  return 48 <= e && e <= 57 || 65 <= e && e <= 70 || 97 <= e && e <= 102;
}
p(xc, "isHexCode");
function bc(e) {
  return 48 <= e && e <= 55;
}
p(bc, "isOctCode");
function Cc(e) {
  return 48 <= e && e <= 57;
}
p(Cc, "isDecCode");
function kc(e) {
  if (e === null) return !1;
  var t = e.length, r = 0, i = !1, a;
  if (!t) return !1;
  if (a = e[r], (a === "-" || a === "+") && (a = e[++r]), a === "0") {
    if (r + 1 === t) return !0;
    if (a = e[++r], a === "b") {
      for (r++; r < t; r++)
        if (a = e[r], a !== "_") {
          if (a !== "0" && a !== "1") return !1;
          i = !0;
        }
      return i && a !== "_";
    }
    if (a === "x") {
      for (r++; r < t; r++)
        if (a = e[r], a !== "_") {
          if (!xc(e.charCodeAt(r))) return !1;
          i = !0;
        }
      return i && a !== "_";
    }
    if (a === "o") {
      for (r++; r < t; r++)
        if (a = e[r], a !== "_") {
          if (!bc(e.charCodeAt(r))) return !1;
          i = !0;
        }
      return i && a !== "_";
    }
  }
  if (a === "_") return !1;
  for (; r < t; r++)
    if (a = e[r], a !== "_") {
      if (!Cc(e.charCodeAt(r)))
        return !1;
      i = !0;
    }
  return !(!i || a === "_");
}
p(kc, "resolveYamlInteger");
function wc(e) {
  var t = e, r = 1, i;
  if (t.indexOf("_") !== -1 && (t = t.replace(/_/g, "")), i = t[0], (i === "-" || i === "+") && (i === "-" && (r = -1), t = t.slice(1), i = t[0]), t === "0") return 0;
  if (i === "0") {
    if (t[1] === "b") return r * parseInt(t.slice(2), 2);
    if (t[1] === "x") return r * parseInt(t.slice(2), 16);
    if (t[1] === "o") return r * parseInt(t.slice(2), 8);
  }
  return r * parseInt(t, 10);
}
p(wc, "constructYamlInteger");
function _c(e) {
  return Object.prototype.toString.call(e) === "[object Number]" && e % 1 === 0 && !xt.isNegativeZero(e);
}
p(_c, "isInteger");
var x0 = new Bt("tag:yaml.org,2002:int", {
  kind: "scalar",
  resolve: kc,
  construct: wc,
  predicate: _c,
  represent: {
    binary: /* @__PURE__ */ p(function(e) {
      return e >= 0 ? "0b" + e.toString(2) : "-0b" + e.toString(2).slice(1);
    }, "binary"),
    octal: /* @__PURE__ */ p(function(e) {
      return e >= 0 ? "0o" + e.toString(8) : "-0o" + e.toString(8).slice(1);
    }, "octal"),
    decimal: /* @__PURE__ */ p(function(e) {
      return e.toString(10);
    }, "decimal"),
    /* eslint-disable max-len */
    hexadecimal: /* @__PURE__ */ p(function(e) {
      return e >= 0 ? "0x" + e.toString(16).toUpperCase() : "-0x" + e.toString(16).toUpperCase().slice(1);
    }, "hexadecimal")
  },
  defaultStyle: "decimal",
  styleAliases: {
    binary: [2, "bin"],
    octal: [8, "oct"],
    decimal: [10, "dec"],
    hexadecimal: [16, "hex"]
  }
}), b0 = new RegExp(
  // 2.5e4, 2.5 and integers
  "^(?:[-+]?(?:[0-9][0-9_]*)(?:\\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?|\\.[0-9_]+(?:[eE][-+]?[0-9]+)?|[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$"
);
function vc(e) {
  return !(e === null || !b0.test(e) || // Quick hack to not allow integers end with `_`
  // Probably should update regexp & check speed
  e[e.length - 1] === "_");
}
p(vc, "resolveYamlFloat");
function Sc(e) {
  var t, r;
  return t = e.replace(/_/g, "").toLowerCase(), r = t[0] === "-" ? -1 : 1, "+-".indexOf(t[0]) >= 0 && (t = t.slice(1)), t === ".inf" ? r === 1 ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY : t === ".nan" ? NaN : r * parseFloat(t, 10);
}
p(Sc, "constructYamlFloat");
var C0 = /^[-+]?[0-9]+e/;
function Tc(e, t) {
  var r;
  if (isNaN(e))
    switch (t) {
      case "lowercase":
        return ".nan";
      case "uppercase":
        return ".NAN";
      case "camelcase":
        return ".NaN";
    }
  else if (Number.POSITIVE_INFINITY === e)
    switch (t) {
      case "lowercase":
        return ".inf";
      case "uppercase":
        return ".INF";
      case "camelcase":
        return ".Inf";
    }
  else if (Number.NEGATIVE_INFINITY === e)
    switch (t) {
      case "lowercase":
        return "-.inf";
      case "uppercase":
        return "-.INF";
      case "camelcase":
        return "-.Inf";
    }
  else if (xt.isNegativeZero(e))
    return "-0.0";
  return r = e.toString(10), C0.test(r) ? r.replace("e", ".e") : r;
}
p(Tc, "representYamlFloat");
function Bc(e) {
  return Object.prototype.toString.call(e) === "[object Number]" && (e % 1 !== 0 || xt.isNegativeZero(e));
}
p(Bc, "isFloat");
var k0 = new Bt("tag:yaml.org,2002:float", {
  kind: "scalar",
  resolve: vc,
  construct: Sc,
  predicate: Bc,
  represent: Tc,
  defaultStyle: "lowercase"
}), Lc = g0.extend({
  implicit: [
    m0,
    y0,
    x0,
    k0
  ]
}), w0 = Lc, Mc = new RegExp(
  "^([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])$"
), $c = new RegExp(
  "^([0-9][0-9][0-9][0-9])-([0-9][0-9]?)-([0-9][0-9]?)(?:[Tt]|[ \\t]+)([0-9][0-9]?):([0-9][0-9]):([0-9][0-9])(?:\\.([0-9]*))?(?:[ \\t]*(Z|([-+])([0-9][0-9]?)(?::([0-9][0-9]))?))?$"
);
function Ac(e) {
  return e === null ? !1 : Mc.exec(e) !== null || $c.exec(e) !== null;
}
p(Ac, "resolveYamlTimestamp");
function Fc(e) {
  var t, r, i, a, n, o, s, l = 0, c = null, h, u, f;
  if (t = Mc.exec(e), t === null && (t = $c.exec(e)), t === null) throw new Error("Date resolve error");
  if (r = +t[1], i = +t[2] - 1, a = +t[3], !t[4])
    return new Date(Date.UTC(r, i, a));
  if (n = +t[4], o = +t[5], s = +t[6], t[7]) {
    for (l = t[7].slice(0, 3); l.length < 3; )
      l += "0";
    l = +l;
  }
  return t[9] && (h = +t[10], u = +(t[11] || 0), c = (h * 60 + u) * 6e4, t[9] === "-" && (c = -c)), f = new Date(Date.UTC(r, i, a, n, o, s, l)), c && f.setTime(f.getTime() - c), f;
}
p(Fc, "constructYamlTimestamp");
function Ec(e) {
  return e.toISOString();
}
p(Ec, "representYamlTimestamp");
var _0 = new Bt("tag:yaml.org,2002:timestamp", {
  kind: "scalar",
  resolve: Ac,
  construct: Fc,
  instanceOf: Date,
  represent: Ec
});
function Oc(e) {
  return e === "<<" || e === null;
}
p(Oc, "resolveYamlMerge");
var v0 = new Bt("tag:yaml.org,2002:merge", {
  kind: "scalar",
  resolve: Oc
}), bs = `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=
\r`;
function Dc(e) {
  if (e === null) return !1;
  var t, r, i = 0, a = e.length, n = bs;
  for (r = 0; r < a; r++)
    if (t = n.indexOf(e.charAt(r)), !(t > 64)) {
      if (t < 0) return !1;
      i += 6;
    }
  return i % 8 === 0;
}
p(Dc, "resolveYamlBinary");
function Rc(e) {
  var t, r, i = e.replace(/[\r\n=]/g, ""), a = i.length, n = bs, o = 0, s = [];
  for (t = 0; t < a; t++)
    t % 4 === 0 && t && (s.push(o >> 16 & 255), s.push(o >> 8 & 255), s.push(o & 255)), o = o << 6 | n.indexOf(i.charAt(t));
  return r = a % 4 * 6, r === 0 ? (s.push(o >> 16 & 255), s.push(o >> 8 & 255), s.push(o & 255)) : r === 18 ? (s.push(o >> 10 & 255), s.push(o >> 2 & 255)) : r === 12 && s.push(o >> 4 & 255), new Uint8Array(s);
}
p(Rc, "constructYamlBinary");
function Pc(e) {
  var t = "", r = 0, i, a, n = e.length, o = bs;
  for (i = 0; i < n; i++)
    i % 3 === 0 && i && (t += o[r >> 18 & 63], t += o[r >> 12 & 63], t += o[r >> 6 & 63], t += o[r & 63]), r = (r << 8) + e[i];
  return a = n % 3, a === 0 ? (t += o[r >> 18 & 63], t += o[r >> 12 & 63], t += o[r >> 6 & 63], t += o[r & 63]) : a === 2 ? (t += o[r >> 10 & 63], t += o[r >> 4 & 63], t += o[r << 2 & 63], t += o[64]) : a === 1 && (t += o[r >> 2 & 63], t += o[r << 4 & 63], t += o[64], t += o[64]), t;
}
p(Pc, "representYamlBinary");
function Ic(e) {
  return Object.prototype.toString.call(e) === "[object Uint8Array]";
}
p(Ic, "isBinary");
var S0 = new Bt("tag:yaml.org,2002:binary", {
  kind: "scalar",
  resolve: Dc,
  construct: Rc,
  predicate: Ic,
  represent: Pc
}), T0 = Object.prototype.hasOwnProperty, B0 = Object.prototype.toString;
function Nc(e) {
  if (e === null) return !0;
  var t = [], r, i, a, n, o, s = e;
  for (r = 0, i = s.length; r < i; r += 1) {
    if (a = s[r], o = !1, B0.call(a) !== "[object Object]") return !1;
    for (n in a)
      if (T0.call(a, n))
        if (!o) o = !0;
        else return !1;
    if (!o) return !1;
    if (t.indexOf(n) === -1) t.push(n);
    else return !1;
  }
  return !0;
}
p(Nc, "resolveYamlOmap");
function zc(e) {
  return e !== null ? e : [];
}
p(zc, "constructYamlOmap");
var L0 = new Bt("tag:yaml.org,2002:omap", {
  kind: "sequence",
  resolve: Nc,
  construct: zc
}), M0 = Object.prototype.toString;
function qc(e) {
  if (e === null) return !0;
  var t, r, i, a, n, o = e;
  for (n = new Array(o.length), t = 0, r = o.length; t < r; t += 1) {
    if (i = o[t], M0.call(i) !== "[object Object]" || (a = Object.keys(i), a.length !== 1)) return !1;
    n[t] = [a[0], i[a[0]]];
  }
  return !0;
}
p(qc, "resolveYamlPairs");
function Wc(e) {
  if (e === null) return [];
  var t, r, i, a, n, o = e;
  for (n = new Array(o.length), t = 0, r = o.length; t < r; t += 1)
    i = o[t], a = Object.keys(i), n[t] = [a[0], i[a[0]]];
  return n;
}
p(Wc, "constructYamlPairs");
var $0 = new Bt("tag:yaml.org,2002:pairs", {
  kind: "sequence",
  resolve: qc,
  construct: Wc
}), A0 = Object.prototype.hasOwnProperty;
function Hc(e) {
  if (e === null) return !0;
  var t, r = e;
  for (t in r)
    if (A0.call(r, t) && r[t] !== null)
      return !1;
  return !0;
}
p(Hc, "resolveYamlSet");
function jc(e) {
  return e !== null ? e : {};
}
p(jc, "constructYamlSet");
var F0 = new Bt("tag:yaml.org,2002:set", {
  kind: "mapping",
  resolve: Hc,
  construct: jc
}), Yc = w0.extend({
  implicit: [
    _0,
    v0
  ],
  explicit: [
    S0,
    L0,
    $0,
    F0
  ]
}), _e = Object.prototype.hasOwnProperty, Xi = 1, Gc = 2, Uc = 3, Vi = 4, Qa = 1, E0 = 2, xo = 3, O0 = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uFFFE\uFFFF]|[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?:[^\uD800-\uDBFF]|^)[\uDC00-\uDFFF]/, D0 = /[\x85\u2028\u2029]/, R0 = /[,\[\]\{\}]/, Xc = /^(?:!|!!|![a-z\-]+!)$/i, Vc = /^(?:!|[^,\[\]\{\}])(?:%[0-9a-f]{2}|[0-9a-z\-#;\/\?:@&=\+\$,_\.!~\*'\(\)\[\]])*$/i;
function kn(e) {
  return Object.prototype.toString.call(e);
}
p(kn, "_class");
function Gt(e) {
  return e === 10 || e === 13;
}
p(Gt, "is_EOL");
function we(e) {
  return e === 9 || e === 32;
}
p(we, "is_WHITE_SPACE");
function At(e) {
  return e === 9 || e === 32 || e === 10 || e === 13;
}
p(At, "is_WS_OR_EOL");
function Oe(e) {
  return e === 44 || e === 91 || e === 93 || e === 123 || e === 125;
}
p(Oe, "is_FLOW_INDICATOR");
function Zc(e) {
  var t;
  return 48 <= e && e <= 57 ? e - 48 : (t = e | 32, 97 <= t && t <= 102 ? t - 97 + 10 : -1);
}
p(Zc, "fromHexCode");
function Kc(e) {
  return e === 120 ? 2 : e === 117 ? 4 : e === 85 ? 8 : 0;
}
p(Kc, "escapedHexLen");
function Qc(e) {
  return 48 <= e && e <= 57 ? e - 48 : -1;
}
p(Qc, "fromDecimalCode");
function wn(e) {
  return e === 48 ? "\0" : e === 97 ? "\x07" : e === 98 ? "\b" : e === 116 || e === 9 ? "	" : e === 110 ? `
` : e === 118 ? "\v" : e === 102 ? "\f" : e === 114 ? "\r" : e === 101 ? "\x1B" : e === 32 ? " " : e === 34 ? '"' : e === 47 ? "/" : e === 92 ? "\\" : e === 78 ? "" : e === 95 ? " " : e === 76 ? "\u2028" : e === 80 ? "\u2029" : "";
}
p(wn, "simpleEscapeSequence");
function Jc(e) {
  return e <= 65535 ? String.fromCharCode(e) : String.fromCharCode(
    (e - 65536 >> 10) + 55296,
    (e - 65536 & 1023) + 56320
  );
}
p(Jc, "charFromCodepoint");
var th = new Array(256), eh = new Array(256);
for (Me = 0; Me < 256; Me++)
  th[Me] = wn(Me) ? 1 : 0, eh[Me] = wn(Me);
var Me;
function rh(e, t) {
  this.input = e, this.filename = t.filename || null, this.schema = t.schema || Yc, this.onWarning = t.onWarning || null, this.legacy = t.legacy || !1, this.json = t.json || !1, this.listener = t.listener || null, this.implicitTypes = this.schema.compiledImplicit, this.typeMap = this.schema.compiledTypeMap, this.length = e.length, this.position = 0, this.line = 0, this.lineStart = 0, this.lineIndent = 0, this.firstTabInLine = -1, this.documents = [];
}
p(rh, "State$1");
function Cs(e, t) {
  var r = {
    name: e.filename,
    buffer: e.input.slice(0, -1),
    // omit trailing \0
    position: e.position,
    line: e.line,
    column: e.position - e.lineStart
  };
  return r.snippet = l0(r), new Rt(t, r);
}
p(Cs, "generateError");
function G(e, t) {
  throw Cs(e, t);
}
p(G, "throwError");
function Kr(e, t) {
  e.onWarning && e.onWarning.call(null, Cs(e, t));
}
p(Kr, "throwWarning");
var bo = {
  YAML: /* @__PURE__ */ p(function(t, r, i) {
    var a, n, o;
    t.version !== null && G(t, "duplication of %YAML directive"), i.length !== 1 && G(t, "YAML directive accepts exactly one argument"), a = /^([0-9]+)\.([0-9]+)$/.exec(i[0]), a === null && G(t, "ill-formed argument of the YAML directive"), n = parseInt(a[1], 10), o = parseInt(a[2], 10), n !== 1 && G(t, "unacceptable YAML version of the document"), t.version = i[0], t.checkLineBreaks = o < 2, o !== 1 && o !== 2 && Kr(t, "unsupported YAML version of the document");
  }, "handleYamlDirective"),
  TAG: /* @__PURE__ */ p(function(t, r, i) {
    var a, n;
    i.length !== 2 && G(t, "TAG directive accepts exactly two arguments"), a = i[0], n = i[1], Xc.test(a) || G(t, "ill-formed tag handle (first argument) of the TAG directive"), _e.call(t.tagMap, a) && G(t, 'there is a previously declared suffix for "' + a + '" tag handle'), Vc.test(n) || G(t, "ill-formed tag prefix (second argument) of the TAG directive");
    try {
      n = decodeURIComponent(n);
    } catch {
      G(t, "tag prefix is malformed: " + n);
    }
    t.tagMap[a] = n;
  }, "handleTagDirective")
};
function fe(e, t, r, i) {
  var a, n, o, s;
  if (t < r) {
    if (s = e.input.slice(t, r), i)
      for (a = 0, n = s.length; a < n; a += 1)
        o = s.charCodeAt(a), o === 9 || 32 <= o && o <= 1114111 || G(e, "expected valid JSON character");
    else O0.test(s) && G(e, "the stream contains non-printable characters");
    e.result += s;
  }
}
p(fe, "captureSegment");
function _n(e, t, r, i) {
  var a, n, o, s;
  for (xt.isObject(r) || G(e, "cannot merge mappings; the provided source object is unacceptable"), a = Object.keys(r), o = 0, s = a.length; o < s; o += 1)
    n = a[o], _e.call(t, n) || (t[n] = r[n], i[n] = !0);
}
p(_n, "mergeMappings");
function De(e, t, r, i, a, n, o, s, l) {
  var c, h;
  if (Array.isArray(a))
    for (a = Array.prototype.slice.call(a), c = 0, h = a.length; c < h; c += 1)
      Array.isArray(a[c]) && G(e, "nested arrays are not supported inside keys"), typeof a == "object" && kn(a[c]) === "[object Object]" && (a[c] = "[object Object]");
  if (typeof a == "object" && kn(a) === "[object Object]" && (a = "[object Object]"), a = String(a), t === null && (t = {}), i === "tag:yaml.org,2002:merge")
    if (Array.isArray(n))
      for (c = 0, h = n.length; c < h; c += 1)
        _n(e, t, n[c], r);
    else
      _n(e, t, n, r);
  else
    !e.json && !_e.call(r, a) && _e.call(t, a) && (e.line = o || e.line, e.lineStart = s || e.lineStart, e.position = l || e.position, G(e, "duplicated mapping key")), a === "__proto__" ? Object.defineProperty(t, a, {
      configurable: !0,
      enumerable: !0,
      writable: !0,
      value: n
    }) : t[a] = n, delete r[a];
  return t;
}
p(De, "storeMappingPair");
function Ma(e) {
  var t;
  t = e.input.charCodeAt(e.position), t === 10 ? e.position++ : t === 13 ? (e.position++, e.input.charCodeAt(e.position) === 10 && e.position++) : G(e, "a line break is expected"), e.line += 1, e.lineStart = e.position, e.firstTabInLine = -1;
}
p(Ma, "readLineBreak");
function pt(e, t, r) {
  for (var i = 0, a = e.input.charCodeAt(e.position); a !== 0; ) {
    for (; we(a); )
      a === 9 && e.firstTabInLine === -1 && (e.firstTabInLine = e.position), a = e.input.charCodeAt(++e.position);
    if (t && a === 35)
      do
        a = e.input.charCodeAt(++e.position);
      while (a !== 10 && a !== 13 && a !== 0);
    if (Gt(a))
      for (Ma(e), a = e.input.charCodeAt(e.position), i++, e.lineIndent = 0; a === 32; )
        e.lineIndent++, a = e.input.charCodeAt(++e.position);
    else
      break;
  }
  return r !== -1 && i !== 0 && e.lineIndent < r && Kr(e, "deficient indentation"), i;
}
p(pt, "skipSeparationSpace");
function ui(e) {
  var t = e.position, r;
  return r = e.input.charCodeAt(t), !!((r === 45 || r === 46) && r === e.input.charCodeAt(t + 1) && r === e.input.charCodeAt(t + 2) && (t += 3, r = e.input.charCodeAt(t), r === 0 || At(r)));
}
p(ui, "testDocumentSeparator");
function $a(e, t) {
  t === 1 ? e.result += " " : t > 1 && (e.result += xt.repeat(`
`, t - 1));
}
p($a, "writeFoldedLines");
function ih(e, t, r) {
  var i, a, n, o, s, l, c, h, u = e.kind, f = e.result, d;
  if (d = e.input.charCodeAt(e.position), At(d) || Oe(d) || d === 35 || d === 38 || d === 42 || d === 33 || d === 124 || d === 62 || d === 39 || d === 34 || d === 37 || d === 64 || d === 96 || (d === 63 || d === 45) && (a = e.input.charCodeAt(e.position + 1), At(a) || r && Oe(a)))
    return !1;
  for (e.kind = "scalar", e.result = "", n = o = e.position, s = !1; d !== 0; ) {
    if (d === 58) {
      if (a = e.input.charCodeAt(e.position + 1), At(a) || r && Oe(a))
        break;
    } else if (d === 35) {
      if (i = e.input.charCodeAt(e.position - 1), At(i))
        break;
    } else {
      if (e.position === e.lineStart && ui(e) || r && Oe(d))
        break;
      if (Gt(d))
        if (l = e.line, c = e.lineStart, h = e.lineIndent, pt(e, !1, -1), e.lineIndent >= t) {
          s = !0, d = e.input.charCodeAt(e.position);
          continue;
        } else {
          e.position = o, e.line = l, e.lineStart = c, e.lineIndent = h;
          break;
        }
    }
    s && (fe(e, n, o, !1), $a(e, e.line - l), n = o = e.position, s = !1), we(d) || (o = e.position + 1), d = e.input.charCodeAt(++e.position);
  }
  return fe(e, n, o, !1), e.result ? !0 : (e.kind = u, e.result = f, !1);
}
p(ih, "readPlainScalar");
function ah(e, t) {
  var r, i, a;
  if (r = e.input.charCodeAt(e.position), r !== 39)
    return !1;
  for (e.kind = "scalar", e.result = "", e.position++, i = a = e.position; (r = e.input.charCodeAt(e.position)) !== 0; )
    if (r === 39)
      if (fe(e, i, e.position, !0), r = e.input.charCodeAt(++e.position), r === 39)
        i = e.position, e.position++, a = e.position;
      else
        return !0;
    else Gt(r) ? (fe(e, i, a, !0), $a(e, pt(e, !1, t)), i = a = e.position) : e.position === e.lineStart && ui(e) ? G(e, "unexpected end of the document within a single quoted scalar") : (e.position++, a = e.position);
  G(e, "unexpected end of the stream within a single quoted scalar");
}
p(ah, "readSingleQuotedScalar");
function nh(e, t) {
  var r, i, a, n, o, s;
  if (s = e.input.charCodeAt(e.position), s !== 34)
    return !1;
  for (e.kind = "scalar", e.result = "", e.position++, r = i = e.position; (s = e.input.charCodeAt(e.position)) !== 0; ) {
    if (s === 34)
      return fe(e, r, e.position, !0), e.position++, !0;
    if (s === 92) {
      if (fe(e, r, e.position, !0), s = e.input.charCodeAt(++e.position), Gt(s))
        pt(e, !1, t);
      else if (s < 256 && th[s])
        e.result += eh[s], e.position++;
      else if ((o = Kc(s)) > 0) {
        for (a = o, n = 0; a > 0; a--)
          s = e.input.charCodeAt(++e.position), (o = Zc(s)) >= 0 ? n = (n << 4) + o : G(e, "expected hexadecimal character");
        e.result += Jc(n), e.position++;
      } else
        G(e, "unknown escape sequence");
      r = i = e.position;
    } else Gt(s) ? (fe(e, r, i, !0), $a(e, pt(e, !1, t)), r = i = e.position) : e.position === e.lineStart && ui(e) ? G(e, "unexpected end of the document within a double quoted scalar") : (e.position++, i = e.position);
  }
  G(e, "unexpected end of the stream within a double quoted scalar");
}
p(nh, "readDoubleQuotedScalar");
function sh(e, t) {
  var r = !0, i, a, n, o = e.tag, s, l = e.anchor, c, h, u, f, d, g = /* @__PURE__ */ Object.create(null), m, y, x, b;
  if (b = e.input.charCodeAt(e.position), b === 91)
    h = 93, d = !1, s = [];
  else if (b === 123)
    h = 125, d = !0, s = {};
  else
    return !1;
  for (e.anchor !== null && (e.anchorMap[e.anchor] = s), b = e.input.charCodeAt(++e.position); b !== 0; ) {
    if (pt(e, !0, t), b = e.input.charCodeAt(e.position), b === h)
      return e.position++, e.tag = o, e.anchor = l, e.kind = d ? "mapping" : "sequence", e.result = s, !0;
    r ? b === 44 && G(e, "expected the node content, but found ','") : G(e, "missed comma between flow collection entries"), y = m = x = null, u = f = !1, b === 63 && (c = e.input.charCodeAt(e.position + 1), At(c) && (u = f = !0, e.position++, pt(e, !0, t))), i = e.line, a = e.lineStart, n = e.position, We(e, t, Xi, !1, !0), y = e.tag, m = e.result, pt(e, !0, t), b = e.input.charCodeAt(e.position), (f || e.line === i) && b === 58 && (u = !0, b = e.input.charCodeAt(++e.position), pt(e, !0, t), We(e, t, Xi, !1, !0), x = e.result), d ? De(e, s, g, y, m, x, i, a, n) : u ? s.push(De(e, null, g, y, m, x, i, a, n)) : s.push(m), pt(e, !0, t), b = e.input.charCodeAt(e.position), b === 44 ? (r = !0, b = e.input.charCodeAt(++e.position)) : r = !1;
  }
  G(e, "unexpected end of the stream within a flow collection");
}
p(sh, "readFlowCollection");
function oh(e, t) {
  var r, i, a = Qa, n = !1, o = !1, s = t, l = 0, c = !1, h, u;
  if (u = e.input.charCodeAt(e.position), u === 124)
    i = !1;
  else if (u === 62)
    i = !0;
  else
    return !1;
  for (e.kind = "scalar", e.result = ""; u !== 0; )
    if (u = e.input.charCodeAt(++e.position), u === 43 || u === 45)
      Qa === a ? a = u === 43 ? xo : E0 : G(e, "repeat of a chomping mode identifier");
    else if ((h = Qc(u)) >= 0)
      h === 0 ? G(e, "bad explicit indentation width of a block scalar; it cannot be less than one") : o ? G(e, "repeat of an indentation width identifier") : (s = t + h - 1, o = !0);
    else
      break;
  if (we(u)) {
    do
      u = e.input.charCodeAt(++e.position);
    while (we(u));
    if (u === 35)
      do
        u = e.input.charCodeAt(++e.position);
      while (!Gt(u) && u !== 0);
  }
  for (; u !== 0; ) {
    for (Ma(e), e.lineIndent = 0, u = e.input.charCodeAt(e.position); (!o || e.lineIndent < s) && u === 32; )
      e.lineIndent++, u = e.input.charCodeAt(++e.position);
    if (!o && e.lineIndent > s && (s = e.lineIndent), Gt(u)) {
      l++;
      continue;
    }
    if (e.lineIndent < s) {
      a === xo ? e.result += xt.repeat(`
`, n ? 1 + l : l) : a === Qa && n && (e.result += `
`);
      break;
    }
    for (i ? we(u) ? (c = !0, e.result += xt.repeat(`
`, n ? 1 + l : l)) : c ? (c = !1, e.result += xt.repeat(`
`, l + 1)) : l === 0 ? n && (e.result += " ") : e.result += xt.repeat(`
`, l) : e.result += xt.repeat(`
`, n ? 1 + l : l), n = !0, o = !0, l = 0, r = e.position; !Gt(u) && u !== 0; )
      u = e.input.charCodeAt(++e.position);
    fe(e, r, e.position, !1);
  }
  return !0;
}
p(oh, "readBlockScalar");
function vn(e, t) {
  var r, i = e.tag, a = e.anchor, n = [], o, s = !1, l;
  if (e.firstTabInLine !== -1) return !1;
  for (e.anchor !== null && (e.anchorMap[e.anchor] = n), l = e.input.charCodeAt(e.position); l !== 0 && (e.firstTabInLine !== -1 && (e.position = e.firstTabInLine, G(e, "tab characters must not be used in indentation")), !(l !== 45 || (o = e.input.charCodeAt(e.position + 1), !At(o)))); ) {
    if (s = !0, e.position++, pt(e, !0, -1) && e.lineIndent <= t) {
      n.push(null), l = e.input.charCodeAt(e.position);
      continue;
    }
    if (r = e.line, We(e, t, Uc, !1, !0), n.push(e.result), pt(e, !0, -1), l = e.input.charCodeAt(e.position), (e.line === r || e.lineIndent > t) && l !== 0)
      G(e, "bad indentation of a sequence entry");
    else if (e.lineIndent < t)
      break;
  }
  return s ? (e.tag = i, e.anchor = a, e.kind = "sequence", e.result = n, !0) : !1;
}
p(vn, "readBlockSequence");
function lh(e, t, r) {
  var i, a, n, o, s, l, c = e.tag, h = e.anchor, u = {}, f = /* @__PURE__ */ Object.create(null), d = null, g = null, m = null, y = !1, x = !1, b;
  if (e.firstTabInLine !== -1) return !1;
  for (e.anchor !== null && (e.anchorMap[e.anchor] = u), b = e.input.charCodeAt(e.position); b !== 0; ) {
    if (!y && e.firstTabInLine !== -1 && (e.position = e.firstTabInLine, G(e, "tab characters must not be used in indentation")), i = e.input.charCodeAt(e.position + 1), n = e.line, (b === 63 || b === 58) && At(i))
      b === 63 ? (y && (De(e, u, f, d, g, null, o, s, l), d = g = m = null), x = !0, y = !0, a = !0) : y ? (y = !1, a = !0) : G(e, "incomplete explicit mapping pair; a key node is missed; or followed by a non-tabulated empty line"), e.position += 1, b = i;
    else {
      if (o = e.line, s = e.lineStart, l = e.position, !We(e, r, Gc, !1, !0))
        break;
      if (e.line === n) {
        for (b = e.input.charCodeAt(e.position); we(b); )
          b = e.input.charCodeAt(++e.position);
        if (b === 58)
          b = e.input.charCodeAt(++e.position), At(b) || G(e, "a whitespace character is expected after the key-value separator within a block mapping"), y && (De(e, u, f, d, g, null, o, s, l), d = g = m = null), x = !0, y = !1, a = !1, d = e.tag, g = e.result;
        else if (x)
          G(e, "can not read an implicit mapping pair; a colon is missed");
        else
          return e.tag = c, e.anchor = h, !0;
      } else if (x)
        G(e, "can not read a block mapping entry; a multiline key may not be an implicit key");
      else
        return e.tag = c, e.anchor = h, !0;
    }
    if ((e.line === n || e.lineIndent > t) && (y && (o = e.line, s = e.lineStart, l = e.position), We(e, t, Vi, !0, a) && (y ? g = e.result : m = e.result), y || (De(e, u, f, d, g, m, o, s, l), d = g = m = null), pt(e, !0, -1), b = e.input.charCodeAt(e.position)), (e.line === n || e.lineIndent > t) && b !== 0)
      G(e, "bad indentation of a mapping entry");
    else if (e.lineIndent < t)
      break;
  }
  return y && De(e, u, f, d, g, null, o, s, l), x && (e.tag = c, e.anchor = h, e.kind = "mapping", e.result = u), x;
}
p(lh, "readBlockMapping");
function ch(e) {
  var t, r = !1, i = !1, a, n, o;
  if (o = e.input.charCodeAt(e.position), o !== 33) return !1;
  if (e.tag !== null && G(e, "duplication of a tag property"), o = e.input.charCodeAt(++e.position), o === 60 ? (r = !0, o = e.input.charCodeAt(++e.position)) : o === 33 ? (i = !0, a = "!!", o = e.input.charCodeAt(++e.position)) : a = "!", t = e.position, r) {
    do
      o = e.input.charCodeAt(++e.position);
    while (o !== 0 && o !== 62);
    e.position < e.length ? (n = e.input.slice(t, e.position), o = e.input.charCodeAt(++e.position)) : G(e, "unexpected end of the stream within a verbatim tag");
  } else {
    for (; o !== 0 && !At(o); )
      o === 33 && (i ? G(e, "tag suffix cannot contain exclamation marks") : (a = e.input.slice(t - 1, e.position + 1), Xc.test(a) || G(e, "named tag handle cannot contain such characters"), i = !0, t = e.position + 1)), o = e.input.charCodeAt(++e.position);
    n = e.input.slice(t, e.position), R0.test(n) && G(e, "tag suffix cannot contain flow indicator characters");
  }
  n && !Vc.test(n) && G(e, "tag name cannot contain such characters: " + n);
  try {
    n = decodeURIComponent(n);
  } catch {
    G(e, "tag name is malformed: " + n);
  }
  return r ? e.tag = n : _e.call(e.tagMap, a) ? e.tag = e.tagMap[a] + n : a === "!" ? e.tag = "!" + n : a === "!!" ? e.tag = "tag:yaml.org,2002:" + n : G(e, 'undeclared tag handle "' + a + '"'), !0;
}
p(ch, "readTagProperty");
function hh(e) {
  var t, r;
  if (r = e.input.charCodeAt(e.position), r !== 38) return !1;
  for (e.anchor !== null && G(e, "duplication of an anchor property"), r = e.input.charCodeAt(++e.position), t = e.position; r !== 0 && !At(r) && !Oe(r); )
    r = e.input.charCodeAt(++e.position);
  return e.position === t && G(e, "name of an anchor node must contain at least one character"), e.anchor = e.input.slice(t, e.position), !0;
}
p(hh, "readAnchorProperty");
function uh(e) {
  var t, r, i;
  if (i = e.input.charCodeAt(e.position), i !== 42) return !1;
  for (i = e.input.charCodeAt(++e.position), t = e.position; i !== 0 && !At(i) && !Oe(i); )
    i = e.input.charCodeAt(++e.position);
  return e.position === t && G(e, "name of an alias node must contain at least one character"), r = e.input.slice(t, e.position), _e.call(e.anchorMap, r) || G(e, 'unidentified alias "' + r + '"'), e.result = e.anchorMap[r], pt(e, !0, -1), !0;
}
p(uh, "readAlias");
function We(e, t, r, i, a) {
  var n, o, s, l = 1, c = !1, h = !1, u, f, d, g, m, y;
  if (e.listener !== null && e.listener("open", e), e.tag = null, e.anchor = null, e.kind = null, e.result = null, n = o = s = Vi === r || Uc === r, i && pt(e, !0, -1) && (c = !0, e.lineIndent > t ? l = 1 : e.lineIndent === t ? l = 0 : e.lineIndent < t && (l = -1)), l === 1)
    for (; ch(e) || hh(e); )
      pt(e, !0, -1) ? (c = !0, s = n, e.lineIndent > t ? l = 1 : e.lineIndent === t ? l = 0 : e.lineIndent < t && (l = -1)) : s = !1;
  if (s && (s = c || a), (l === 1 || Vi === r) && (Xi === r || Gc === r ? m = t : m = t + 1, y = e.position - e.lineStart, l === 1 ? s && (vn(e, y) || lh(e, y, m)) || sh(e, m) ? h = !0 : (o && oh(e, m) || ah(e, m) || nh(e, m) ? h = !0 : uh(e) ? (h = !0, (e.tag !== null || e.anchor !== null) && G(e, "alias node should not have any properties")) : ih(e, m, Xi === r) && (h = !0, e.tag === null && (e.tag = "?")), e.anchor !== null && (e.anchorMap[e.anchor] = e.result)) : l === 0 && (h = s && vn(e, y))), e.tag === null)
    e.anchor !== null && (e.anchorMap[e.anchor] = e.result);
  else if (e.tag === "?") {
    for (e.result !== null && e.kind !== "scalar" && G(e, 'unacceptable node kind for !<?> tag; it should be "scalar", not "' + e.kind + '"'), u = 0, f = e.implicitTypes.length; u < f; u += 1)
      if (g = e.implicitTypes[u], g.resolve(e.result)) {
        e.result = g.construct(e.result), e.tag = g.tag, e.anchor !== null && (e.anchorMap[e.anchor] = e.result);
        break;
      }
  } else if (e.tag !== "!") {
    if (_e.call(e.typeMap[e.kind || "fallback"], e.tag))
      g = e.typeMap[e.kind || "fallback"][e.tag];
    else
      for (g = null, d = e.typeMap.multi[e.kind || "fallback"], u = 0, f = d.length; u < f; u += 1)
        if (e.tag.slice(0, d[u].tag.length) === d[u].tag) {
          g = d[u];
          break;
        }
    g || G(e, "unknown tag !<" + e.tag + ">"), e.result !== null && g.kind !== e.kind && G(e, "unacceptable node kind for !<" + e.tag + '> tag; it should be "' + g.kind + '", not "' + e.kind + '"'), g.resolve(e.result, e.tag) ? (e.result = g.construct(e.result, e.tag), e.anchor !== null && (e.anchorMap[e.anchor] = e.result)) : G(e, "cannot resolve a node with !<" + e.tag + "> explicit tag");
  }
  return e.listener !== null && e.listener("close", e), e.tag !== null || e.anchor !== null || h;
}
p(We, "composeNode");
function fh(e) {
  var t = e.position, r, i, a, n = !1, o;
  for (e.version = null, e.checkLineBreaks = e.legacy, e.tagMap = /* @__PURE__ */ Object.create(null), e.anchorMap = /* @__PURE__ */ Object.create(null); (o = e.input.charCodeAt(e.position)) !== 0 && (pt(e, !0, -1), o = e.input.charCodeAt(e.position), !(e.lineIndent > 0 || o !== 37)); ) {
    for (n = !0, o = e.input.charCodeAt(++e.position), r = e.position; o !== 0 && !At(o); )
      o = e.input.charCodeAt(++e.position);
    for (i = e.input.slice(r, e.position), a = [], i.length < 1 && G(e, "directive name must not be less than one character in length"); o !== 0; ) {
      for (; we(o); )
        o = e.input.charCodeAt(++e.position);
      if (o === 35) {
        do
          o = e.input.charCodeAt(++e.position);
        while (o !== 0 && !Gt(o));
        break;
      }
      if (Gt(o)) break;
      for (r = e.position; o !== 0 && !At(o); )
        o = e.input.charCodeAt(++e.position);
      a.push(e.input.slice(r, e.position));
    }
    o !== 0 && Ma(e), _e.call(bo, i) ? bo[i](e, i, a) : Kr(e, 'unknown document directive "' + i + '"');
  }
  if (pt(e, !0, -1), e.lineIndent === 0 && e.input.charCodeAt(e.position) === 45 && e.input.charCodeAt(e.position + 1) === 45 && e.input.charCodeAt(e.position + 2) === 45 ? (e.position += 3, pt(e, !0, -1)) : n && G(e, "directives end mark is expected"), We(e, e.lineIndent - 1, Vi, !1, !0), pt(e, !0, -1), e.checkLineBreaks && D0.test(e.input.slice(t, e.position)) && Kr(e, "non-ASCII line breaks are interpreted as content"), e.documents.push(e.result), e.position === e.lineStart && ui(e)) {
    e.input.charCodeAt(e.position) === 46 && (e.position += 3, pt(e, !0, -1));
    return;
  }
  if (e.position < e.length - 1)
    G(e, "end of the stream or a document separator is expected");
  else
    return;
}
p(fh, "readDocument");
function ks(e, t) {
  e = String(e), t = t || {}, e.length !== 0 && (e.charCodeAt(e.length - 1) !== 10 && e.charCodeAt(e.length - 1) !== 13 && (e += `
`), e.charCodeAt(0) === 65279 && (e = e.slice(1)));
  var r = new rh(e, t), i = e.indexOf("\0");
  for (i !== -1 && (r.position = i, G(r, "null byte is not allowed in input")), r.input += "\0"; r.input.charCodeAt(r.position) === 32; )
    r.lineIndent += 1, r.position += 1;
  for (; r.position < r.length - 1; )
    fh(r);
  return r.documents;
}
p(ks, "loadDocuments");
function P0(e, t, r) {
  t !== null && typeof t == "object" && typeof r > "u" && (r = t, t = null);
  var i = ks(e, r);
  if (typeof t != "function")
    return i;
  for (var a = 0, n = i.length; a < n; a += 1)
    t(i[a]);
}
p(P0, "loadAll$1");
function dh(e, t) {
  var r = ks(e, t);
  if (r.length !== 0) {
    if (r.length === 1)
      return r[0];
    throw new Rt("expected a single document in the stream, but found more");
  }
}
p(dh, "load$1");
var I0 = dh, N0 = {
  load: I0
}, ph = Object.prototype.toString, gh = Object.prototype.hasOwnProperty, ws = 65279, z0 = 9, Qr = 10, q0 = 13, W0 = 32, H0 = 33, j0 = 34, Sn = 35, Y0 = 37, G0 = 38, U0 = 39, X0 = 42, mh = 44, V0 = 45, Zi = 58, Z0 = 61, K0 = 62, Q0 = 63, J0 = 64, yh = 91, xh = 93, tm = 96, bh = 123, em = 124, Ch = 125, Lt = {};
Lt[0] = "\\0";
Lt[7] = "\\a";
Lt[8] = "\\b";
Lt[9] = "\\t";
Lt[10] = "\\n";
Lt[11] = "\\v";
Lt[12] = "\\f";
Lt[13] = "\\r";
Lt[27] = "\\e";
Lt[34] = '\\"';
Lt[92] = "\\\\";
Lt[133] = "\\N";
Lt[160] = "\\_";
Lt[8232] = "\\L";
Lt[8233] = "\\P";
var rm = [
  "y",
  "Y",
  "yes",
  "Yes",
  "YES",
  "on",
  "On",
  "ON",
  "n",
  "N",
  "no",
  "No",
  "NO",
  "off",
  "Off",
  "OFF"
], im = /^[-+]?[0-9_]+(?::[0-9_]+)+(?:\.[0-9_]*)?$/;
function kh(e, t) {
  var r, i, a, n, o, s, l;
  if (t === null) return {};
  for (r = {}, i = Object.keys(t), a = 0, n = i.length; a < n; a += 1)
    o = i[a], s = String(t[o]), o.slice(0, 2) === "!!" && (o = "tag:yaml.org,2002:" + o.slice(2)), l = e.compiledTypeMap.fallback[o], l && gh.call(l.styleAliases, s) && (s = l.styleAliases[s]), r[o] = s;
  return r;
}
p(kh, "compileStyleMap");
function wh(e) {
  var t, r, i;
  if (t = e.toString(16).toUpperCase(), e <= 255)
    r = "x", i = 2;
  else if (e <= 65535)
    r = "u", i = 4;
  else if (e <= 4294967295)
    r = "U", i = 8;
  else
    throw new Rt("code point within a string may not be greater than 0xFFFFFFFF");
  return "\\" + r + xt.repeat("0", i - t.length) + t;
}
p(wh, "encodeHex");
var am = 1, Jr = 2;
function _h(e) {
  this.schema = e.schema || Yc, this.indent = Math.max(1, e.indent || 2), this.noArrayIndent = e.noArrayIndent || !1, this.skipInvalid = e.skipInvalid || !1, this.flowLevel = xt.isNothing(e.flowLevel) ? -1 : e.flowLevel, this.styleMap = kh(this.schema, e.styles || null), this.sortKeys = e.sortKeys || !1, this.lineWidth = e.lineWidth || 80, this.noRefs = e.noRefs || !1, this.noCompatMode = e.noCompatMode || !1, this.condenseFlow = e.condenseFlow || !1, this.quotingType = e.quotingType === '"' ? Jr : am, this.forceQuotes = e.forceQuotes || !1, this.replacer = typeof e.replacer == "function" ? e.replacer : null, this.implicitTypes = this.schema.compiledImplicit, this.explicitTypes = this.schema.compiledExplicit, this.tag = null, this.result = "", this.duplicates = [], this.usedDuplicates = null;
}
p(_h, "State");
function Tn(e, t) {
  for (var r = xt.repeat(" ", t), i = 0, a = -1, n = "", o, s = e.length; i < s; )
    a = e.indexOf(`
`, i), a === -1 ? (o = e.slice(i), i = s) : (o = e.slice(i, a + 1), i = a + 1), o.length && o !== `
` && (n += r), n += o;
  return n;
}
p(Tn, "indentString");
function Ki(e, t) {
  return `
` + xt.repeat(" ", e.indent * t);
}
p(Ki, "generateNextLine");
function vh(e, t) {
  var r, i, a;
  for (r = 0, i = e.implicitTypes.length; r < i; r += 1)
    if (a = e.implicitTypes[r], a.resolve(t))
      return !0;
  return !1;
}
p(vh, "testImplicitResolving");
function ti(e) {
  return e === W0 || e === z0;
}
p(ti, "isWhitespace");
function br(e) {
  return 32 <= e && e <= 126 || 161 <= e && e <= 55295 && e !== 8232 && e !== 8233 || 57344 <= e && e <= 65533 && e !== ws || 65536 <= e && e <= 1114111;
}
p(br, "isPrintable");
function Bn(e) {
  return br(e) && e !== ws && e !== q0 && e !== Qr;
}
p(Bn, "isNsCharOrWhitespace");
function Ln(e, t, r) {
  var i = Bn(e), a = i && !ti(e);
  return (
    // ns-plain-safe
    (r ? (
      // c = flow-in
      i
    ) : i && e !== mh && e !== yh && e !== xh && e !== bh && e !== Ch) && e !== Sn && !(t === Zi && !a) || Bn(t) && !ti(t) && e === Sn || t === Zi && a
  );
}
p(Ln, "isPlainSafe");
function Sh(e) {
  return br(e) && e !== ws && !ti(e) && e !== V0 && e !== Q0 && e !== Zi && e !== mh && e !== yh && e !== xh && e !== bh && e !== Ch && e !== Sn && e !== G0 && e !== X0 && e !== H0 && e !== em && e !== Z0 && e !== K0 && e !== U0 && e !== j0 && e !== Y0 && e !== J0 && e !== tm;
}
p(Sh, "isPlainSafeFirst");
function Th(e) {
  return !ti(e) && e !== Zi;
}
p(Th, "isPlainSafeLast");
function rr(e, t) {
  var r = e.charCodeAt(t), i;
  return r >= 55296 && r <= 56319 && t + 1 < e.length && (i = e.charCodeAt(t + 1), i >= 56320 && i <= 57343) ? (r - 55296) * 1024 + i - 56320 + 65536 : r;
}
p(rr, "codePointAt");
function _s(e) {
  var t = /^\n* /;
  return t.test(e);
}
p(_s, "needIndentIndicator");
var Bh = 1, Mn = 2, Lh = 3, Mh = 4, tr = 5;
function $h(e, t, r, i, a, n, o, s) {
  var l, c = 0, h = null, u = !1, f = !1, d = i !== -1, g = -1, m = Sh(rr(e, 0)) && Th(rr(e, e.length - 1));
  if (t || o)
    for (l = 0; l < e.length; c >= 65536 ? l += 2 : l++) {
      if (c = rr(e, l), !br(c))
        return tr;
      m = m && Ln(c, h, s), h = c;
    }
  else {
    for (l = 0; l < e.length; c >= 65536 ? l += 2 : l++) {
      if (c = rr(e, l), c === Qr)
        u = !0, d && (f = f || // Foldable line = too long, and not more-indented.
        l - g - 1 > i && e[g + 1] !== " ", g = l);
      else if (!br(c))
        return tr;
      m = m && Ln(c, h, s), h = c;
    }
    f = f || d && l - g - 1 > i && e[g + 1] !== " ";
  }
  return !u && !f ? m && !o && !a(e) ? Bh : n === Jr ? tr : Mn : r > 9 && _s(e) ? tr : o ? n === Jr ? tr : Mn : f ? Mh : Lh;
}
p($h, "chooseScalarStyle");
function Ah(e, t, r, i, a) {
  e.dump = function() {
    if (t.length === 0)
      return e.quotingType === Jr ? '""' : "''";
    if (!e.noCompatMode && (rm.indexOf(t) !== -1 || im.test(t)))
      return e.quotingType === Jr ? '"' + t + '"' : "'" + t + "'";
    var n = e.indent * Math.max(1, r), o = e.lineWidth === -1 ? -1 : Math.max(Math.min(e.lineWidth, 40), e.lineWidth - n), s = i || e.flowLevel > -1 && r >= e.flowLevel;
    function l(c) {
      return vh(e, c);
    }
    switch (p(l, "testAmbiguity"), $h(
      t,
      s,
      e.indent,
      o,
      l,
      e.quotingType,
      e.forceQuotes && !i,
      a
    )) {
      case Bh:
        return t;
      case Mn:
        return "'" + t.replace(/'/g, "''") + "'";
      case Lh:
        return "|" + $n(t, e.indent) + An(Tn(t, n));
      case Mh:
        return ">" + $n(t, e.indent) + An(Tn(Fh(t, o), n));
      case tr:
        return '"' + Eh(t) + '"';
      default:
        throw new Rt("impossible error: invalid scalar style");
    }
  }();
}
p(Ah, "writeScalar");
function $n(e, t) {
  var r = _s(e) ? String(t) : "", i = e[e.length - 1] === `
`, a = i && (e[e.length - 2] === `
` || e === `
`), n = a ? "+" : i ? "" : "-";
  return r + n + `
`;
}
p($n, "blockHeader");
function An(e) {
  return e[e.length - 1] === `
` ? e.slice(0, -1) : e;
}
p(An, "dropEndingNewline");
function Fh(e, t) {
  for (var r = /(\n+)([^\n]*)/g, i = function() {
    var c = e.indexOf(`
`);
    return c = c !== -1 ? c : e.length, r.lastIndex = c, Fn(e.slice(0, c), t);
  }(), a = e[0] === `
` || e[0] === " ", n, o; o = r.exec(e); ) {
    var s = o[1], l = o[2];
    n = l[0] === " ", i += s + (!a && !n && l !== "" ? `
` : "") + Fn(l, t), a = n;
  }
  return i;
}
p(Fh, "foldString");
function Fn(e, t) {
  if (e === "" || e[0] === " ") return e;
  for (var r = / [^ ]/g, i, a = 0, n, o = 0, s = 0, l = ""; i = r.exec(e); )
    s = i.index, s - a > t && (n = o > a ? o : s, l += `
` + e.slice(a, n), a = n + 1), o = s;
  return l += `
`, e.length - a > t && o > a ? l += e.slice(a, o) + `
` + e.slice(o + 1) : l += e.slice(a), l.slice(1);
}
p(Fn, "foldLine");
function Eh(e) {
  for (var t = "", r = 0, i, a = 0; a < e.length; r >= 65536 ? a += 2 : a++)
    r = rr(e, a), i = Lt[r], !i && br(r) ? (t += e[a], r >= 65536 && (t += e[a + 1])) : t += i || wh(r);
  return t;
}
p(Eh, "escapeString");
function Oh(e, t, r) {
  var i = "", a = e.tag, n, o, s;
  for (n = 0, o = r.length; n < o; n += 1)
    s = r[n], e.replacer && (s = e.replacer.call(r, String(n), s)), (re(e, t, s, !1, !1) || typeof s > "u" && re(e, t, null, !1, !1)) && (i !== "" && (i += "," + (e.condenseFlow ? "" : " ")), i += e.dump);
  e.tag = a, e.dump = "[" + i + "]";
}
p(Oh, "writeFlowSequence");
function En(e, t, r, i) {
  var a = "", n = e.tag, o, s, l;
  for (o = 0, s = r.length; o < s; o += 1)
    l = r[o], e.replacer && (l = e.replacer.call(r, String(o), l)), (re(e, t + 1, l, !0, !0, !1, !0) || typeof l > "u" && re(e, t + 1, null, !0, !0, !1, !0)) && ((!i || a !== "") && (a += Ki(e, t)), e.dump && Qr === e.dump.charCodeAt(0) ? a += "-" : a += "- ", a += e.dump);
  e.tag = n, e.dump = a || "[]";
}
p(En, "writeBlockSequence");
function Dh(e, t, r) {
  var i = "", a = e.tag, n = Object.keys(r), o, s, l, c, h;
  for (o = 0, s = n.length; o < s; o += 1)
    h = "", i !== "" && (h += ", "), e.condenseFlow && (h += '"'), l = n[o], c = r[l], e.replacer && (c = e.replacer.call(r, l, c)), re(e, t, l, !1, !1) && (e.dump.length > 1024 && (h += "? "), h += e.dump + (e.condenseFlow ? '"' : "") + ":" + (e.condenseFlow ? "" : " "), re(e, t, c, !1, !1) && (h += e.dump, i += h));
  e.tag = a, e.dump = "{" + i + "}";
}
p(Dh, "writeFlowMapping");
function Rh(e, t, r, i) {
  var a = "", n = e.tag, o = Object.keys(r), s, l, c, h, u, f;
  if (e.sortKeys === !0)
    o.sort();
  else if (typeof e.sortKeys == "function")
    o.sort(e.sortKeys);
  else if (e.sortKeys)
    throw new Rt("sortKeys must be a boolean or a function");
  for (s = 0, l = o.length; s < l; s += 1)
    f = "", (!i || a !== "") && (f += Ki(e, t)), c = o[s], h = r[c], e.replacer && (h = e.replacer.call(r, c, h)), re(e, t + 1, c, !0, !0, !0) && (u = e.tag !== null && e.tag !== "?" || e.dump && e.dump.length > 1024, u && (e.dump && Qr === e.dump.charCodeAt(0) ? f += "?" : f += "? "), f += e.dump, u && (f += Ki(e, t)), re(e, t + 1, h, !0, u) && (e.dump && Qr === e.dump.charCodeAt(0) ? f += ":" : f += ": ", f += e.dump, a += f));
  e.tag = n, e.dump = a || "{}";
}
p(Rh, "writeBlockMapping");
function On(e, t, r) {
  var i, a, n, o, s, l;
  for (a = r ? e.explicitTypes : e.implicitTypes, n = 0, o = a.length; n < o; n += 1)
    if (s = a[n], (s.instanceOf || s.predicate) && (!s.instanceOf || typeof t == "object" && t instanceof s.instanceOf) && (!s.predicate || s.predicate(t))) {
      if (r ? s.multi && s.representName ? e.tag = s.representName(t) : e.tag = s.tag : e.tag = "?", s.represent) {
        if (l = e.styleMap[s.tag] || s.defaultStyle, ph.call(s.represent) === "[object Function]")
          i = s.represent(t, l);
        else if (gh.call(s.represent, l))
          i = s.represent[l](t, l);
        else
          throw new Rt("!<" + s.tag + '> tag resolver accepts not "' + l + '" style');
        e.dump = i;
      }
      return !0;
    }
  return !1;
}
p(On, "detectType");
function re(e, t, r, i, a, n, o) {
  e.tag = null, e.dump = r, On(e, r, !1) || On(e, r, !0);
  var s = ph.call(e.dump), l = i, c;
  i && (i = e.flowLevel < 0 || e.flowLevel > t);
  var h = s === "[object Object]" || s === "[object Array]", u, f;
  if (h && (u = e.duplicates.indexOf(r), f = u !== -1), (e.tag !== null && e.tag !== "?" || f || e.indent !== 2 && t > 0) && (a = !1), f && e.usedDuplicates[u])
    e.dump = "*ref_" + u;
  else {
    if (h && f && !e.usedDuplicates[u] && (e.usedDuplicates[u] = !0), s === "[object Object]")
      i && Object.keys(e.dump).length !== 0 ? (Rh(e, t, e.dump, a), f && (e.dump = "&ref_" + u + e.dump)) : (Dh(e, t, e.dump), f && (e.dump = "&ref_" + u + " " + e.dump));
    else if (s === "[object Array]")
      i && e.dump.length !== 0 ? (e.noArrayIndent && !o && t > 0 ? En(e, t - 1, e.dump, a) : En(e, t, e.dump, a), f && (e.dump = "&ref_" + u + e.dump)) : (Oh(e, t, e.dump), f && (e.dump = "&ref_" + u + " " + e.dump));
    else if (s === "[object String]")
      e.tag !== "?" && Ah(e, e.dump, t, n, l);
    else {
      if (s === "[object Undefined]")
        return !1;
      if (e.skipInvalid) return !1;
      throw new Rt("unacceptable kind of an object to dump " + s);
    }
    e.tag !== null && e.tag !== "?" && (c = encodeURI(
      e.tag[0] === "!" ? e.tag.slice(1) : e.tag
    ).replace(/!/g, "%21"), e.tag[0] === "!" ? c = "!" + c : c.slice(0, 18) === "tag:yaml.org,2002:" ? c = "!!" + c.slice(18) : c = "!<" + c + ">", e.dump = c + " " + e.dump);
  }
  return !0;
}
p(re, "writeNode");
function Ph(e, t) {
  var r = [], i = [], a, n;
  for (Qi(e, r, i), a = 0, n = i.length; a < n; a += 1)
    t.duplicates.push(r[i[a]]);
  t.usedDuplicates = new Array(n);
}
p(Ph, "getDuplicateReferences");
function Qi(e, t, r) {
  var i, a, n;
  if (e !== null && typeof e == "object")
    if (a = t.indexOf(e), a !== -1)
      r.indexOf(a) === -1 && r.push(a);
    else if (t.push(e), Array.isArray(e))
      for (a = 0, n = e.length; a < n; a += 1)
        Qi(e[a], t, r);
    else
      for (i = Object.keys(e), a = 0, n = i.length; a < n; a += 1)
        Qi(e[i[a]], t, r);
}
p(Qi, "inspectNode");
function nm(e, t) {
  t = t || {};
  var r = new _h(t);
  r.noRefs || Ph(e, r);
  var i = e;
  return r.replacer && (i = r.replacer.call({ "": i }, "", i)), re(r, 0, i, !0, !0) ? r.dump + `
` : "";
}
p(nm, "dump$1");
function sm(e, t) {
  return function() {
    throw new Error("Function yaml." + e + " is removed in js-yaml 4. Use yaml." + t + " instead, which is now safe by default.");
  };
}
p(sm, "renamed");
var om = Lc, lm = N0.load;
/*! Bundled license information:

js-yaml/dist/js-yaml.mjs:
  (*! js-yaml 4.1.0 https://github.com/nodeca/js-yaml @license MIT *)
*/
var Ht = {
  aggregation: 18,
  extension: 18,
  composition: 18,
  dependency: 6,
  lollipop: 13.5,
  arrow_point: 4
};
function Nr(e, t) {
  if (e === void 0 || t === void 0)
    return { angle: 0, deltaX: 0, deltaY: 0 };
  e = ht(e), t = ht(t);
  const [r, i] = [e.x, e.y], [a, n] = [t.x, t.y], o = a - r, s = n - i;
  return { angle: Math.atan(s / o), deltaX: o, deltaY: s };
}
p(Nr, "calculateDeltaAndAngle");
var ht = /* @__PURE__ */ p((e) => Array.isArray(e) ? { x: e[0], y: e[1] } : e, "pointTransformer"), cm = /* @__PURE__ */ p((e) => ({
  x: /* @__PURE__ */ p(function(t, r, i) {
    let a = 0;
    const n = ht(i[0]).x < ht(i[i.length - 1]).x ? "left" : "right";
    if (r === 0 && Object.hasOwn(Ht, e.arrowTypeStart)) {
      const { angle: d, deltaX: g } = Nr(i[0], i[1]);
      a = Ht[e.arrowTypeStart] * Math.cos(d) * (g >= 0 ? 1 : -1);
    } else if (r === i.length - 1 && Object.hasOwn(Ht, e.arrowTypeEnd)) {
      const { angle: d, deltaX: g } = Nr(
        i[i.length - 1],
        i[i.length - 2]
      );
      a = Ht[e.arrowTypeEnd] * Math.cos(d) * (g >= 0 ? 1 : -1);
    }
    const o = Math.abs(
      ht(t).x - ht(i[i.length - 1]).x
    ), s = Math.abs(
      ht(t).y - ht(i[i.length - 1]).y
    ), l = Math.abs(ht(t).x - ht(i[0]).x), c = Math.abs(ht(t).y - ht(i[0]).y), h = Ht[e.arrowTypeStart], u = Ht[e.arrowTypeEnd], f = 1;
    if (o < u && o > 0 && s < u) {
      let d = u + f - o;
      d *= n === "right" ? -1 : 1, a -= d;
    }
    if (l < h && l > 0 && c < h) {
      let d = h + f - l;
      d *= n === "right" ? -1 : 1, a += d;
    }
    return ht(t).x + a;
  }, "x"),
  y: /* @__PURE__ */ p(function(t, r, i) {
    let a = 0;
    const n = ht(i[0]).y < ht(i[i.length - 1]).y ? "down" : "up";
    if (r === 0 && Object.hasOwn(Ht, e.arrowTypeStart)) {
      const { angle: d, deltaY: g } = Nr(i[0], i[1]);
      a = Ht[e.arrowTypeStart] * Math.abs(Math.sin(d)) * (g >= 0 ? 1 : -1);
    } else if (r === i.length - 1 && Object.hasOwn(Ht, e.arrowTypeEnd)) {
      const { angle: d, deltaY: g } = Nr(
        i[i.length - 1],
        i[i.length - 2]
      );
      a = Ht[e.arrowTypeEnd] * Math.abs(Math.sin(d)) * (g >= 0 ? 1 : -1);
    }
    const o = Math.abs(
      ht(t).y - ht(i[i.length - 1]).y
    ), s = Math.abs(
      ht(t).x - ht(i[i.length - 1]).x
    ), l = Math.abs(ht(t).y - ht(i[0]).y), c = Math.abs(ht(t).x - ht(i[0]).x), h = Ht[e.arrowTypeStart], u = Ht[e.arrowTypeEnd], f = 1;
    if (o < u && o > 0 && s < u) {
      let d = u + f - o;
      d *= n === "up" ? -1 : 1, a -= d;
    }
    if (l < h && l > 0 && c < h) {
      let d = h + f - l;
      d *= n === "up" ? -1 : 1, a += d;
    }
    return ht(t).y + a;
  }, "y")
}), "getLineFunctionsWithOffset"), vs = /* @__PURE__ */ p(({
  flowchart: e
}) => {
  var a, n;
  const t = ((a = e == null ? void 0 : e.subGraphTitleMargin) == null ? void 0 : a.top) ?? 0, r = ((n = e == null ? void 0 : e.subGraphTitleMargin) == null ? void 0 : n.bottom) ?? 0, i = t + r;
  return {
    subGraphTitleTopMargin: t,
    subGraphTitleBottomMargin: r,
    subGraphTitleTotalMargin: i
  };
}, "getSubGraphTitleMargins"), hm = /* @__PURE__ */ p((e) => {
  const { handDrawnSeed: t } = at();
  return {
    fill: e,
    hachureAngle: 120,
    // angle of hachure,
    hachureGap: 4,
    fillWeight: 2,
    roughness: 0.7,
    stroke: e,
    seed: t
  };
}, "solidStateFill"), Sr = /* @__PURE__ */ p((e) => {
  const t = um([...e.cssCompiledStyles || [], ...e.cssStyles || []]);
  return { stylesMap: t, stylesArray: [...t] };
}, "compileStyles"), um = /* @__PURE__ */ p((e) => {
  const t = /* @__PURE__ */ new Map();
  return e.forEach((r) => {
    const [i, a] = r.split(":");
    t.set(i.trim(), a == null ? void 0 : a.trim());
  }), t;
}, "styles2Map"), Ih = /* @__PURE__ */ p((e) => e === "color" || e === "font-size" || e === "font-family" || e === "font-weight" || e === "font-style" || e === "text-decoration" || e === "text-align" || e === "text-transform" || e === "line-height" || e === "letter-spacing" || e === "word-spacing" || e === "text-shadow" || e === "text-overflow" || e === "white-space" || e === "word-wrap" || e === "word-break" || e === "overflow-wrap" || e === "hyphens", "isLabelStyle"), Y = /* @__PURE__ */ p((e) => {
  const { stylesArray: t } = Sr(e), r = [], i = [], a = [], n = [];
  return t.forEach((o) => {
    const s = o[0];
    Ih(s) ? r.push(o.join(":") + " !important") : (i.push(o.join(":") + " !important"), s.includes("stroke") && a.push(o.join(":") + " !important"), s === "fill" && n.push(o.join(":") + " !important"));
  }), {
    labelStyles: r.join(";"),
    nodeStyles: i.join(";"),
    stylesArray: t,
    borderStyles: a,
    backgroundStyles: n
  };
}, "styles2String"), H = /* @__PURE__ */ p((e, t) => {
  var l;
  const { themeVariables: r, handDrawnSeed: i } = at(), { nodeBorder: a, mainBkg: n } = r, { stylesMap: o } = Sr(e);
  return Object.assign(
    {
      roughness: 0.7,
      fill: o.get("fill") || n,
      fillStyle: "hachure",
      // solid fill
      fillWeight: 4,
      hachureGap: 5.2,
      stroke: o.get("stroke") || a,
      seed: i,
      strokeWidth: ((l = o.get("stroke-width")) == null ? void 0 : l.replace("px", "")) || 1.3,
      fillLineDash: [0, 0]
    },
    t
  );
}, "userNodeOverrides"), Ss = {}, Ct = {};
Object.defineProperty(Ct, "__esModule", { value: !0 });
Ct.BLANK_URL = Ct.relativeFirstCharacters = Ct.whitespaceEscapeCharsRegex = Ct.urlSchemeRegex = Ct.ctrlCharactersRegex = Ct.htmlCtrlEntityRegex = Ct.htmlEntitiesRegex = Ct.invalidProtocolRegex = void 0;
Ct.invalidProtocolRegex = /^([^\w]*)(javascript|data|vbscript)/im;
Ct.htmlEntitiesRegex = /&#(\w+)(^\w|;)?/g;
Ct.htmlCtrlEntityRegex = /&(newline|tab);/gi;
Ct.ctrlCharactersRegex = /[\u0000-\u001F\u007F-\u009F\u2000-\u200D\uFEFF]/gim;
Ct.urlSchemeRegex = /^.+(:|&colon;)/gim;
Ct.whitespaceEscapeCharsRegex = /(\\|%5[cC])((%(6[eE]|72|74))|[nrt])/g;
Ct.relativeFirstCharacters = [".", "/"];
Ct.BLANK_URL = "about:blank";
Object.defineProperty(Ss, "__esModule", { value: !0 });
var Nh = Ss.sanitizeUrl = void 0, Tt = Ct;
function fm(e) {
  return Tt.relativeFirstCharacters.indexOf(e[0]) > -1;
}
function dm(e) {
  var t = e.replace(Tt.ctrlCharactersRegex, "");
  return t.replace(Tt.htmlEntitiesRegex, function(r, i) {
    return String.fromCharCode(i);
  });
}
function pm(e) {
  return URL.canParse(e);
}
function Co(e) {
  try {
    return decodeURIComponent(e);
  } catch {
    return e;
  }
}
function gm(e) {
  if (!e)
    return Tt.BLANK_URL;
  var t, r = Co(e.trim());
  do
    r = dm(r).replace(Tt.htmlCtrlEntityRegex, "").replace(Tt.ctrlCharactersRegex, "").replace(Tt.whitespaceEscapeCharsRegex, "").trim(), r = Co(r), t = r.match(Tt.ctrlCharactersRegex) || r.match(Tt.htmlEntitiesRegex) || r.match(Tt.htmlCtrlEntityRegex) || r.match(Tt.whitespaceEscapeCharsRegex);
  while (t && t.length > 0);
  var i = r;
  if (!i)
    return Tt.BLANK_URL;
  if (fm(i))
    return i;
  var a = i.trimStart(), n = a.match(Tt.urlSchemeRegex);
  if (!n)
    return i;
  var o = n[0].toLowerCase().trim();
  if (Tt.invalidProtocolRegex.test(o))
    return Tt.BLANK_URL;
  var s = a.replace(/\\/g, "/");
  if (o === "mailto:" || o.includes("://"))
    return s;
  if (o === "http:" || o === "https:") {
    if (!pm(s))
      return Tt.BLANK_URL;
    var l = new URL(s);
    return l.protocol = l.protocol.toLowerCase(), l.hostname = l.hostname.toLowerCase(), l.toString();
  }
  return s;
}
Nh = Ss.sanitizeUrl = gm;
var mm = { value: () => {
} };
function zh() {
  for (var e = 0, t = arguments.length, r = {}, i; e < t; ++e) {
    if (!(i = arguments[e] + "") || i in r || /[\s.]/.test(i)) throw new Error("illegal type: " + i);
    r[i] = [];
  }
  return new Ei(r);
}
function Ei(e) {
  this._ = e;
}
function ym(e, t) {
  return e.trim().split(/^|\s+/).map(function(r) {
    var i = "", a = r.indexOf(".");
    if (a >= 0 && (i = r.slice(a + 1), r = r.slice(0, a)), r && !t.hasOwnProperty(r)) throw new Error("unknown type: " + r);
    return { type: r, name: i };
  });
}
Ei.prototype = zh.prototype = {
  constructor: Ei,
  on: function(e, t) {
    var r = this._, i = ym(e + "", r), a, n = -1, o = i.length;
    if (arguments.length < 2) {
      for (; ++n < o; ) if ((a = (e = i[n]).type) && (a = xm(r[a], e.name))) return a;
      return;
    }
    if (t != null && typeof t != "function") throw new Error("invalid callback: " + t);
    for (; ++n < o; )
      if (a = (e = i[n]).type) r[a] = ko(r[a], e.name, t);
      else if (t == null) for (a in r) r[a] = ko(r[a], e.name, null);
    return this;
  },
  copy: function() {
    var e = {}, t = this._;
    for (var r in t) e[r] = t[r].slice();
    return new Ei(e);
  },
  call: function(e, t) {
    if ((a = arguments.length - 2) > 0) for (var r = new Array(a), i = 0, a, n; i < a; ++i) r[i] = arguments[i + 2];
    if (!this._.hasOwnProperty(e)) throw new Error("unknown type: " + e);
    for (n = this._[e], i = 0, a = n.length; i < a; ++i) n[i].value.apply(t, r);
  },
  apply: function(e, t, r) {
    if (!this._.hasOwnProperty(e)) throw new Error("unknown type: " + e);
    for (var i = this._[e], a = 0, n = i.length; a < n; ++a) i[a].value.apply(t, r);
  }
};
function xm(e, t) {
  for (var r = 0, i = e.length, a; r < i; ++r)
    if ((a = e[r]).name === t)
      return a.value;
}
function ko(e, t, r) {
  for (var i = 0, a = e.length; i < a; ++i)
    if (e[i].name === t) {
      e[i] = mm, e = e.slice(0, i).concat(e.slice(i + 1));
      break;
    }
  return r != null && e.push({ name: t, value: r }), e;
}
var Dn = "http://www.w3.org/1999/xhtml";
const wo = {
  svg: "http://www.w3.org/2000/svg",
  xhtml: Dn,
  xlink: "http://www.w3.org/1999/xlink",
  xml: "http://www.w3.org/XML/1998/namespace",
  xmlns: "http://www.w3.org/2000/xmlns/"
};
function Aa(e) {
  var t = e += "", r = t.indexOf(":");
  return r >= 0 && (t = e.slice(0, r)) !== "xmlns" && (e = e.slice(r + 1)), wo.hasOwnProperty(t) ? { space: wo[t], local: e } : e;
}
function bm(e) {
  return function() {
    var t = this.ownerDocument, r = this.namespaceURI;
    return r === Dn && t.documentElement.namespaceURI === Dn ? t.createElement(e) : t.createElementNS(r, e);
  };
}
function Cm(e) {
  return function() {
    return this.ownerDocument.createElementNS(e.space, e.local);
  };
}
function qh(e) {
  var t = Aa(e);
  return (t.local ? Cm : bm)(t);
}
function km() {
}
function Ts(e) {
  return e == null ? km : function() {
    return this.querySelector(e);
  };
}
function wm(e) {
  typeof e != "function" && (e = Ts(e));
  for (var t = this._groups, r = t.length, i = new Array(r), a = 0; a < r; ++a)
    for (var n = t[a], o = n.length, s = i[a] = new Array(o), l, c, h = 0; h < o; ++h)
      (l = n[h]) && (c = e.call(l, l.__data__, h, n)) && ("__data__" in l && (c.__data__ = l.__data__), s[h] = c);
  return new qt(i, this._parents);
}
function _m(e) {
  return e == null ? [] : Array.isArray(e) ? e : Array.from(e);
}
function vm() {
  return [];
}
function Wh(e) {
  return e == null ? vm : function() {
    return this.querySelectorAll(e);
  };
}
function Sm(e) {
  return function() {
    return _m(e.apply(this, arguments));
  };
}
function Tm(e) {
  typeof e == "function" ? e = Sm(e) : e = Wh(e);
  for (var t = this._groups, r = t.length, i = [], a = [], n = 0; n < r; ++n)
    for (var o = t[n], s = o.length, l, c = 0; c < s; ++c)
      (l = o[c]) && (i.push(e.call(l, l.__data__, c, o)), a.push(l));
  return new qt(i, a);
}
function Hh(e) {
  return function() {
    return this.matches(e);
  };
}
function jh(e) {
  return function(t) {
    return t.matches(e);
  };
}
var Bm = Array.prototype.find;
function Lm(e) {
  return function() {
    return Bm.call(this.children, e);
  };
}
function Mm() {
  return this.firstElementChild;
}
function $m(e) {
  return this.select(e == null ? Mm : Lm(typeof e == "function" ? e : jh(e)));
}
var Am = Array.prototype.filter;
function Fm() {
  return Array.from(this.children);
}
function Em(e) {
  return function() {
    return Am.call(this.children, e);
  };
}
function Om(e) {
  return this.selectAll(e == null ? Fm : Em(typeof e == "function" ? e : jh(e)));
}
function Dm(e) {
  typeof e != "function" && (e = Hh(e));
  for (var t = this._groups, r = t.length, i = new Array(r), a = 0; a < r; ++a)
    for (var n = t[a], o = n.length, s = i[a] = [], l, c = 0; c < o; ++c)
      (l = n[c]) && e.call(l, l.__data__, c, n) && s.push(l);
  return new qt(i, this._parents);
}
function Yh(e) {
  return new Array(e.length);
}
function Rm() {
  return new qt(this._enter || this._groups.map(Yh), this._parents);
}
function Ji(e, t) {
  this.ownerDocument = e.ownerDocument, this.namespaceURI = e.namespaceURI, this._next = null, this._parent = e, this.__data__ = t;
}
Ji.prototype = {
  constructor: Ji,
  appendChild: function(e) {
    return this._parent.insertBefore(e, this._next);
  },
  insertBefore: function(e, t) {
    return this._parent.insertBefore(e, t);
  },
  querySelector: function(e) {
    return this._parent.querySelector(e);
  },
  querySelectorAll: function(e) {
    return this._parent.querySelectorAll(e);
  }
};
function Pm(e) {
  return function() {
    return e;
  };
}
function Im(e, t, r, i, a, n) {
  for (var o = 0, s, l = t.length, c = n.length; o < c; ++o)
    (s = t[o]) ? (s.__data__ = n[o], i[o] = s) : r[o] = new Ji(e, n[o]);
  for (; o < l; ++o)
    (s = t[o]) && (a[o] = s);
}
function Nm(e, t, r, i, a, n, o) {
  var s, l, c = /* @__PURE__ */ new Map(), h = t.length, u = n.length, f = new Array(h), d;
  for (s = 0; s < h; ++s)
    (l = t[s]) && (f[s] = d = o.call(l, l.__data__, s, t) + "", c.has(d) ? a[s] = l : c.set(d, l));
  for (s = 0; s < u; ++s)
    d = o.call(e, n[s], s, n) + "", (l = c.get(d)) ? (i[s] = l, l.__data__ = n[s], c.delete(d)) : r[s] = new Ji(e, n[s]);
  for (s = 0; s < h; ++s)
    (l = t[s]) && c.get(f[s]) === l && (a[s] = l);
}
function zm(e) {
  return e.__data__;
}
function qm(e, t) {
  if (!arguments.length) return Array.from(this, zm);
  var r = t ? Nm : Im, i = this._parents, a = this._groups;
  typeof e != "function" && (e = Pm(e));
  for (var n = a.length, o = new Array(n), s = new Array(n), l = new Array(n), c = 0; c < n; ++c) {
    var h = i[c], u = a[c], f = u.length, d = Wm(e.call(h, h && h.__data__, c, i)), g = d.length, m = s[c] = new Array(g), y = o[c] = new Array(g), x = l[c] = new Array(f);
    r(h, u, m, y, x, d, t);
    for (var b = 0, k = 0, S, w; b < g; ++b)
      if (S = m[b]) {
        for (b >= k && (k = b + 1); !(w = y[k]) && ++k < g; ) ;
        S._next = w || null;
      }
  }
  return o = new qt(o, i), o._enter = s, o._exit = l, o;
}
function Wm(e) {
  return typeof e == "object" && "length" in e ? e : Array.from(e);
}
function Hm() {
  return new qt(this._exit || this._groups.map(Yh), this._parents);
}
function jm(e, t, r) {
  var i = this.enter(), a = this, n = this.exit();
  return typeof e == "function" ? (i = e(i), i && (i = i.selection())) : i = i.append(e + ""), t != null && (a = t(a), a && (a = a.selection())), r == null ? n.remove() : r(n), i && a ? i.merge(a).order() : a;
}
function Ym(e) {
  for (var t = e.selection ? e.selection() : e, r = this._groups, i = t._groups, a = r.length, n = i.length, o = Math.min(a, n), s = new Array(a), l = 0; l < o; ++l)
    for (var c = r[l], h = i[l], u = c.length, f = s[l] = new Array(u), d, g = 0; g < u; ++g)
      (d = c[g] || h[g]) && (f[g] = d);
  for (; l < a; ++l)
    s[l] = r[l];
  return new qt(s, this._parents);
}
function Gm() {
  for (var e = this._groups, t = -1, r = e.length; ++t < r; )
    for (var i = e[t], a = i.length - 1, n = i[a], o; --a >= 0; )
      (o = i[a]) && (n && o.compareDocumentPosition(n) ^ 4 && n.parentNode.insertBefore(o, n), n = o);
  return this;
}
function Um(e) {
  e || (e = Xm);
  function t(u, f) {
    return u && f ? e(u.__data__, f.__data__) : !u - !f;
  }
  for (var r = this._groups, i = r.length, a = new Array(i), n = 0; n < i; ++n) {
    for (var o = r[n], s = o.length, l = a[n] = new Array(s), c, h = 0; h < s; ++h)
      (c = o[h]) && (l[h] = c);
    l.sort(t);
  }
  return new qt(a, this._parents).order();
}
function Xm(e, t) {
  return e < t ? -1 : e > t ? 1 : e >= t ? 0 : NaN;
}
function Vm() {
  var e = arguments[0];
  return arguments[0] = this, e.apply(null, arguments), this;
}
function Zm() {
  return Array.from(this);
}
function Km() {
  for (var e = this._groups, t = 0, r = e.length; t < r; ++t)
    for (var i = e[t], a = 0, n = i.length; a < n; ++a) {
      var o = i[a];
      if (o) return o;
    }
  return null;
}
function Qm() {
  let e = 0;
  for (const t of this) ++e;
  return e;
}
function Jm() {
  return !this.node();
}
function ty(e) {
  for (var t = this._groups, r = 0, i = t.length; r < i; ++r)
    for (var a = t[r], n = 0, o = a.length, s; n < o; ++n)
      (s = a[n]) && e.call(s, s.__data__, n, a);
  return this;
}
function ey(e) {
  return function() {
    this.removeAttribute(e);
  };
}
function ry(e) {
  return function() {
    this.removeAttributeNS(e.space, e.local);
  };
}
function iy(e, t) {
  return function() {
    this.setAttribute(e, t);
  };
}
function ay(e, t) {
  return function() {
    this.setAttributeNS(e.space, e.local, t);
  };
}
function ny(e, t) {
  return function() {
    var r = t.apply(this, arguments);
    r == null ? this.removeAttribute(e) : this.setAttribute(e, r);
  };
}
function sy(e, t) {
  return function() {
    var r = t.apply(this, arguments);
    r == null ? this.removeAttributeNS(e.space, e.local) : this.setAttributeNS(e.space, e.local, r);
  };
}
function oy(e, t) {
  var r = Aa(e);
  if (arguments.length < 2) {
    var i = this.node();
    return r.local ? i.getAttributeNS(r.space, r.local) : i.getAttribute(r);
  }
  return this.each((t == null ? r.local ? ry : ey : typeof t == "function" ? r.local ? sy : ny : r.local ? ay : iy)(r, t));
}
function Gh(e) {
  return e.ownerDocument && e.ownerDocument.defaultView || e.document && e || e.defaultView;
}
function ly(e) {
  return function() {
    this.style.removeProperty(e);
  };
}
function cy(e, t, r) {
  return function() {
    this.style.setProperty(e, t, r);
  };
}
function hy(e, t, r) {
  return function() {
    var i = t.apply(this, arguments);
    i == null ? this.style.removeProperty(e) : this.style.setProperty(e, i, r);
  };
}
function uy(e, t, r) {
  return arguments.length > 1 ? this.each((t == null ? ly : typeof t == "function" ? hy : cy)(e, t, r ?? "")) : Cr(this.node(), e);
}
function Cr(e, t) {
  return e.style.getPropertyValue(t) || Gh(e).getComputedStyle(e, null).getPropertyValue(t);
}
function fy(e) {
  return function() {
    delete this[e];
  };
}
function dy(e, t) {
  return function() {
    this[e] = t;
  };
}
function py(e, t) {
  return function() {
    var r = t.apply(this, arguments);
    r == null ? delete this[e] : this[e] = r;
  };
}
function gy(e, t) {
  return arguments.length > 1 ? this.each((t == null ? fy : typeof t == "function" ? py : dy)(e, t)) : this.node()[e];
}
function Uh(e) {
  return e.trim().split(/^|\s+/);
}
function Bs(e) {
  return e.classList || new Xh(e);
}
function Xh(e) {
  this._node = e, this._names = Uh(e.getAttribute("class") || "");
}
Xh.prototype = {
  add: function(e) {
    var t = this._names.indexOf(e);
    t < 0 && (this._names.push(e), this._node.setAttribute("class", this._names.join(" ")));
  },
  remove: function(e) {
    var t = this._names.indexOf(e);
    t >= 0 && (this._names.splice(t, 1), this._node.setAttribute("class", this._names.join(" ")));
  },
  contains: function(e) {
    return this._names.indexOf(e) >= 0;
  }
};
function Vh(e, t) {
  for (var r = Bs(e), i = -1, a = t.length; ++i < a; ) r.add(t[i]);
}
function Zh(e, t) {
  for (var r = Bs(e), i = -1, a = t.length; ++i < a; ) r.remove(t[i]);
}
function my(e) {
  return function() {
    Vh(this, e);
  };
}
function yy(e) {
  return function() {
    Zh(this, e);
  };
}
function xy(e, t) {
  return function() {
    (t.apply(this, arguments) ? Vh : Zh)(this, e);
  };
}
function by(e, t) {
  var r = Uh(e + "");
  if (arguments.length < 2) {
    for (var i = Bs(this.node()), a = -1, n = r.length; ++a < n; ) if (!i.contains(r[a])) return !1;
    return !0;
  }
  return this.each((typeof t == "function" ? xy : t ? my : yy)(r, t));
}
function Cy() {
  this.textContent = "";
}
function ky(e) {
  return function() {
    this.textContent = e;
  };
}
function wy(e) {
  return function() {
    var t = e.apply(this, arguments);
    this.textContent = t ?? "";
  };
}
function _y(e) {
  return arguments.length ? this.each(e == null ? Cy : (typeof e == "function" ? wy : ky)(e)) : this.node().textContent;
}
function vy() {
  this.innerHTML = "";
}
function Sy(e) {
  return function() {
    this.innerHTML = e;
  };
}
function Ty(e) {
  return function() {
    var t = e.apply(this, arguments);
    this.innerHTML = t ?? "";
  };
}
function By(e) {
  return arguments.length ? this.each(e == null ? vy : (typeof e == "function" ? Ty : Sy)(e)) : this.node().innerHTML;
}
function Ly() {
  this.nextSibling && this.parentNode.appendChild(this);
}
function My() {
  return this.each(Ly);
}
function $y() {
  this.previousSibling && this.parentNode.insertBefore(this, this.parentNode.firstChild);
}
function Ay() {
  return this.each($y);
}
function Fy(e) {
  var t = typeof e == "function" ? e : qh(e);
  return this.select(function() {
    return this.appendChild(t.apply(this, arguments));
  });
}
function Ey() {
  return null;
}
function Oy(e, t) {
  var r = typeof e == "function" ? e : qh(e), i = t == null ? Ey : typeof t == "function" ? t : Ts(t);
  return this.select(function() {
    return this.insertBefore(r.apply(this, arguments), i.apply(this, arguments) || null);
  });
}
function Dy() {
  var e = this.parentNode;
  e && e.removeChild(this);
}
function Ry() {
  return this.each(Dy);
}
function Py() {
  var e = this.cloneNode(!1), t = this.parentNode;
  return t ? t.insertBefore(e, this.nextSibling) : e;
}
function Iy() {
  var e = this.cloneNode(!0), t = this.parentNode;
  return t ? t.insertBefore(e, this.nextSibling) : e;
}
function Ny(e) {
  return this.select(e ? Iy : Py);
}
function zy(e) {
  return arguments.length ? this.property("__data__", e) : this.node().__data__;
}
function qy(e) {
  return function(t) {
    e.call(this, t, this.__data__);
  };
}
function Wy(e) {
  return e.trim().split(/^|\s+/).map(function(t) {
    var r = "", i = t.indexOf(".");
    return i >= 0 && (r = t.slice(i + 1), t = t.slice(0, i)), { type: t, name: r };
  });
}
function Hy(e) {
  return function() {
    var t = this.__on;
    if (t) {
      for (var r = 0, i = -1, a = t.length, n; r < a; ++r)
        n = t[r], (!e.type || n.type === e.type) && n.name === e.name ? this.removeEventListener(n.type, n.listener, n.options) : t[++i] = n;
      ++i ? t.length = i : delete this.__on;
    }
  };
}
function jy(e, t, r) {
  return function() {
    var i = this.__on, a, n = qy(t);
    if (i) {
      for (var o = 0, s = i.length; o < s; ++o)
        if ((a = i[o]).type === e.type && a.name === e.name) {
          this.removeEventListener(a.type, a.listener, a.options), this.addEventListener(a.type, a.listener = n, a.options = r), a.value = t;
          return;
        }
    }
    this.addEventListener(e.type, n, r), a = { type: e.type, name: e.name, value: t, listener: n, options: r }, i ? i.push(a) : this.__on = [a];
  };
}
function Yy(e, t, r) {
  var i = Wy(e + ""), a, n = i.length, o;
  if (arguments.length < 2) {
    var s = this.node().__on;
    if (s) {
      for (var l = 0, c = s.length, h; l < c; ++l)
        for (a = 0, h = s[l]; a < n; ++a)
          if ((o = i[a]).type === h.type && o.name === h.name)
            return h.value;
    }
    return;
  }
  for (s = t ? jy : Hy, a = 0; a < n; ++a) this.each(s(i[a], t, r));
  return this;
}
function Kh(e, t, r) {
  var i = Gh(e), a = i.CustomEvent;
  typeof a == "function" ? a = new a(t, r) : (a = i.document.createEvent("Event"), r ? (a.initEvent(t, r.bubbles, r.cancelable), a.detail = r.detail) : a.initEvent(t, !1, !1)), e.dispatchEvent(a);
}
function Gy(e, t) {
  return function() {
    return Kh(this, e, t);
  };
}
function Uy(e, t) {
  return function() {
    return Kh(this, e, t.apply(this, arguments));
  };
}
function Xy(e, t) {
  return this.each((typeof t == "function" ? Uy : Gy)(e, t));
}
function* Vy() {
  for (var e = this._groups, t = 0, r = e.length; t < r; ++t)
    for (var i = e[t], a = 0, n = i.length, o; a < n; ++a)
      (o = i[a]) && (yield o);
}
var Qh = [null];
function qt(e, t) {
  this._groups = e, this._parents = t;
}
function fi() {
  return new qt([[document.documentElement]], Qh);
}
function Zy() {
  return this;
}
qt.prototype = fi.prototype = {
  constructor: qt,
  select: wm,
  selectAll: Tm,
  selectChild: $m,
  selectChildren: Om,
  filter: Dm,
  data: qm,
  enter: Rm,
  exit: Hm,
  join: jm,
  merge: Ym,
  selection: Zy,
  order: Gm,
  sort: Um,
  call: Vm,
  nodes: Zm,
  node: Km,
  size: Qm,
  empty: Jm,
  each: ty,
  attr: oy,
  style: uy,
  property: gy,
  classed: by,
  text: _y,
  html: By,
  raise: My,
  lower: Ay,
  append: Fy,
  insert: Oy,
  remove: Ry,
  clone: Ny,
  datum: zy,
  on: Yy,
  dispatch: Xy,
  [Symbol.iterator]: Vy
};
function et(e) {
  return typeof e == "string" ? new qt([[document.querySelector(e)]], [document.documentElement]) : new qt([[e]], Qh);
}
function Ls(e, t, r) {
  e.prototype = t.prototype = r, r.constructor = e;
}
function Jh(e, t) {
  var r = Object.create(e.prototype);
  for (var i in t) r[i] = t[i];
  return r;
}
function di() {
}
var ei = 0.7, ta = 1 / ei, ar = "\\s*([+-]?\\d+)\\s*", ri = "\\s*([+-]?(?:\\d*\\.)?\\d+(?:[eE][+-]?\\d+)?)\\s*", te = "\\s*([+-]?(?:\\d*\\.)?\\d+(?:[eE][+-]?\\d+)?)%\\s*", Ky = /^#([0-9a-f]{3,8})$/, Qy = new RegExp(`^rgb\\(${ar},${ar},${ar}\\)$`), Jy = new RegExp(`^rgb\\(${te},${te},${te}\\)$`), tx = new RegExp(`^rgba\\(${ar},${ar},${ar},${ri}\\)$`), ex = new RegExp(`^rgba\\(${te},${te},${te},${ri}\\)$`), rx = new RegExp(`^hsl\\(${ri},${te},${te}\\)$`), ix = new RegExp(`^hsla\\(${ri},${te},${te},${ri}\\)$`), _o = {
  aliceblue: 15792383,
  antiquewhite: 16444375,
  aqua: 65535,
  aquamarine: 8388564,
  azure: 15794175,
  beige: 16119260,
  bisque: 16770244,
  black: 0,
  blanchedalmond: 16772045,
  blue: 255,
  blueviolet: 9055202,
  brown: 10824234,
  burlywood: 14596231,
  cadetblue: 6266528,
  chartreuse: 8388352,
  chocolate: 13789470,
  coral: 16744272,
  cornflowerblue: 6591981,
  cornsilk: 16775388,
  crimson: 14423100,
  cyan: 65535,
  darkblue: 139,
  darkcyan: 35723,
  darkgoldenrod: 12092939,
  darkgray: 11119017,
  darkgreen: 25600,
  darkgrey: 11119017,
  darkkhaki: 12433259,
  darkmagenta: 9109643,
  darkolivegreen: 5597999,
  darkorange: 16747520,
  darkorchid: 10040012,
  darkred: 9109504,
  darksalmon: 15308410,
  darkseagreen: 9419919,
  darkslateblue: 4734347,
  darkslategray: 3100495,
  darkslategrey: 3100495,
  darkturquoise: 52945,
  darkviolet: 9699539,
  deeppink: 16716947,
  deepskyblue: 49151,
  dimgray: 6908265,
  dimgrey: 6908265,
  dodgerblue: 2003199,
  firebrick: 11674146,
  floralwhite: 16775920,
  forestgreen: 2263842,
  fuchsia: 16711935,
  gainsboro: 14474460,
  ghostwhite: 16316671,
  gold: 16766720,
  goldenrod: 14329120,
  gray: 8421504,
  green: 32768,
  greenyellow: 11403055,
  grey: 8421504,
  honeydew: 15794160,
  hotpink: 16738740,
  indianred: 13458524,
  indigo: 4915330,
  ivory: 16777200,
  khaki: 15787660,
  lavender: 15132410,
  lavenderblush: 16773365,
  lawngreen: 8190976,
  lemonchiffon: 16775885,
  lightblue: 11393254,
  lightcoral: 15761536,
  lightcyan: 14745599,
  lightgoldenrodyellow: 16448210,
  lightgray: 13882323,
  lightgreen: 9498256,
  lightgrey: 13882323,
  lightpink: 16758465,
  lightsalmon: 16752762,
  lightseagreen: 2142890,
  lightskyblue: 8900346,
  lightslategray: 7833753,
  lightslategrey: 7833753,
  lightsteelblue: 11584734,
  lightyellow: 16777184,
  lime: 65280,
  limegreen: 3329330,
  linen: 16445670,
  magenta: 16711935,
  maroon: 8388608,
  mediumaquamarine: 6737322,
  mediumblue: 205,
  mediumorchid: 12211667,
  mediumpurple: 9662683,
  mediumseagreen: 3978097,
  mediumslateblue: 8087790,
  mediumspringgreen: 64154,
  mediumturquoise: 4772300,
  mediumvioletred: 13047173,
  midnightblue: 1644912,
  mintcream: 16121850,
  mistyrose: 16770273,
  moccasin: 16770229,
  navajowhite: 16768685,
  navy: 128,
  oldlace: 16643558,
  olive: 8421376,
  olivedrab: 7048739,
  orange: 16753920,
  orangered: 16729344,
  orchid: 14315734,
  palegoldenrod: 15657130,
  palegreen: 10025880,
  paleturquoise: 11529966,
  palevioletred: 14381203,
  papayawhip: 16773077,
  peachpuff: 16767673,
  peru: 13468991,
  pink: 16761035,
  plum: 14524637,
  powderblue: 11591910,
  purple: 8388736,
  rebeccapurple: 6697881,
  red: 16711680,
  rosybrown: 12357519,
  royalblue: 4286945,
  saddlebrown: 9127187,
  salmon: 16416882,
  sandybrown: 16032864,
  seagreen: 3050327,
  seashell: 16774638,
  sienna: 10506797,
  silver: 12632256,
  skyblue: 8900331,
  slateblue: 6970061,
  slategray: 7372944,
  slategrey: 7372944,
  snow: 16775930,
  springgreen: 65407,
  steelblue: 4620980,
  tan: 13808780,
  teal: 32896,
  thistle: 14204888,
  tomato: 16737095,
  turquoise: 4251856,
  violet: 15631086,
  wheat: 16113331,
  white: 16777215,
  whitesmoke: 16119285,
  yellow: 16776960,
  yellowgreen: 10145074
};
Ls(di, ii, {
  copy(e) {
    return Object.assign(new this.constructor(), this, e);
  },
  displayable() {
    return this.rgb().displayable();
  },
  hex: vo,
  // Deprecated! Use color.formatHex.
  formatHex: vo,
  formatHex8: ax,
  formatHsl: nx,
  formatRgb: So,
  toString: So
});
function vo() {
  return this.rgb().formatHex();
}
function ax() {
  return this.rgb().formatHex8();
}
function nx() {
  return tu(this).formatHsl();
}
function So() {
  return this.rgb().formatRgb();
}
function ii(e) {
  var t, r;
  return e = (e + "").trim().toLowerCase(), (t = Ky.exec(e)) ? (r = t[1].length, t = parseInt(t[1], 16), r === 6 ? To(t) : r === 3 ? new Pt(t >> 8 & 15 | t >> 4 & 240, t >> 4 & 15 | t & 240, (t & 15) << 4 | t & 15, 1) : r === 8 ? Ci(t >> 24 & 255, t >> 16 & 255, t >> 8 & 255, (t & 255) / 255) : r === 4 ? Ci(t >> 12 & 15 | t >> 8 & 240, t >> 8 & 15 | t >> 4 & 240, t >> 4 & 15 | t & 240, ((t & 15) << 4 | t & 15) / 255) : null) : (t = Qy.exec(e)) ? new Pt(t[1], t[2], t[3], 1) : (t = Jy.exec(e)) ? new Pt(t[1] * 255 / 100, t[2] * 255 / 100, t[3] * 255 / 100, 1) : (t = tx.exec(e)) ? Ci(t[1], t[2], t[3], t[4]) : (t = ex.exec(e)) ? Ci(t[1] * 255 / 100, t[2] * 255 / 100, t[3] * 255 / 100, t[4]) : (t = rx.exec(e)) ? Mo(t[1], t[2] / 100, t[3] / 100, 1) : (t = ix.exec(e)) ? Mo(t[1], t[2] / 100, t[3] / 100, t[4]) : _o.hasOwnProperty(e) ? To(_o[e]) : e === "transparent" ? new Pt(NaN, NaN, NaN, 0) : null;
}
function To(e) {
  return new Pt(e >> 16 & 255, e >> 8 & 255, e & 255, 1);
}
function Ci(e, t, r, i) {
  return i <= 0 && (e = t = r = NaN), new Pt(e, t, r, i);
}
function sx(e) {
  return e instanceof di || (e = ii(e)), e ? (e = e.rgb(), new Pt(e.r, e.g, e.b, e.opacity)) : new Pt();
}
function Rn(e, t, r, i) {
  return arguments.length === 1 ? sx(e) : new Pt(e, t, r, i ?? 1);
}
function Pt(e, t, r, i) {
  this.r = +e, this.g = +t, this.b = +r, this.opacity = +i;
}
Ls(Pt, Rn, Jh(di, {
  brighter(e) {
    return e = e == null ? ta : Math.pow(ta, e), new Pt(this.r * e, this.g * e, this.b * e, this.opacity);
  },
  darker(e) {
    return e = e == null ? ei : Math.pow(ei, e), new Pt(this.r * e, this.g * e, this.b * e, this.opacity);
  },
  rgb() {
    return this;
  },
  clamp() {
    return new Pt(Ie(this.r), Ie(this.g), Ie(this.b), ea(this.opacity));
  },
  displayable() {
    return -0.5 <= this.r && this.r < 255.5 && -0.5 <= this.g && this.g < 255.5 && -0.5 <= this.b && this.b < 255.5 && 0 <= this.opacity && this.opacity <= 1;
  },
  hex: Bo,
  // Deprecated! Use color.formatHex.
  formatHex: Bo,
  formatHex8: ox,
  formatRgb: Lo,
  toString: Lo
}));
function Bo() {
  return `#${Re(this.r)}${Re(this.g)}${Re(this.b)}`;
}
function ox() {
  return `#${Re(this.r)}${Re(this.g)}${Re(this.b)}${Re((isNaN(this.opacity) ? 1 : this.opacity) * 255)}`;
}
function Lo() {
  const e = ea(this.opacity);
  return `${e === 1 ? "rgb(" : "rgba("}${Ie(this.r)}, ${Ie(this.g)}, ${Ie(this.b)}${e === 1 ? ")" : `, ${e})`}`;
}
function ea(e) {
  return isNaN(e) ? 1 : Math.max(0, Math.min(1, e));
}
function Ie(e) {
  return Math.max(0, Math.min(255, Math.round(e) || 0));
}
function Re(e) {
  return e = Ie(e), (e < 16 ? "0" : "") + e.toString(16);
}
function Mo(e, t, r, i) {
  return i <= 0 ? e = t = r = NaN : r <= 0 || r >= 1 ? e = t = NaN : t <= 0 && (e = NaN), new Yt(e, t, r, i);
}
function tu(e) {
  if (e instanceof Yt) return new Yt(e.h, e.s, e.l, e.opacity);
  if (e instanceof di || (e = ii(e)), !e) return new Yt();
  if (e instanceof Yt) return e;
  e = e.rgb();
  var t = e.r / 255, r = e.g / 255, i = e.b / 255, a = Math.min(t, r, i), n = Math.max(t, r, i), o = NaN, s = n - a, l = (n + a) / 2;
  return s ? (t === n ? o = (r - i) / s + (r < i) * 6 : r === n ? o = (i - t) / s + 2 : o = (t - r) / s + 4, s /= l < 0.5 ? n + a : 2 - n - a, o *= 60) : s = l > 0 && l < 1 ? 0 : o, new Yt(o, s, l, e.opacity);
}
function lx(e, t, r, i) {
  return arguments.length === 1 ? tu(e) : new Yt(e, t, r, i ?? 1);
}
function Yt(e, t, r, i) {
  this.h = +e, this.s = +t, this.l = +r, this.opacity = +i;
}
Ls(Yt, lx, Jh(di, {
  brighter(e) {
    return e = e == null ? ta : Math.pow(ta, e), new Yt(this.h, this.s, this.l * e, this.opacity);
  },
  darker(e) {
    return e = e == null ? ei : Math.pow(ei, e), new Yt(this.h, this.s, this.l * e, this.opacity);
  },
  rgb() {
    var e = this.h % 360 + (this.h < 0) * 360, t = isNaN(e) || isNaN(this.s) ? 0 : this.s, r = this.l, i = r + (r < 0.5 ? r : 1 - r) * t, a = 2 * r - i;
    return new Pt(
      Ja(e >= 240 ? e - 240 : e + 120, a, i),
      Ja(e, a, i),
      Ja(e < 120 ? e + 240 : e - 120, a, i),
      this.opacity
    );
  },
  clamp() {
    return new Yt($o(this.h), ki(this.s), ki(this.l), ea(this.opacity));
  },
  displayable() {
    return (0 <= this.s && this.s <= 1 || isNaN(this.s)) && 0 <= this.l && this.l <= 1 && 0 <= this.opacity && this.opacity <= 1;
  },
  formatHsl() {
    const e = ea(this.opacity);
    return `${e === 1 ? "hsl(" : "hsla("}${$o(this.h)}, ${ki(this.s) * 100}%, ${ki(this.l) * 100}%${e === 1 ? ")" : `, ${e})`}`;
  }
}));
function $o(e) {
  return e = (e || 0) % 360, e < 0 ? e + 360 : e;
}
function ki(e) {
  return Math.max(0, Math.min(1, e || 0));
}
function Ja(e, t, r) {
  return (e < 60 ? t + (r - t) * e / 60 : e < 180 ? r : e < 240 ? t + (r - t) * (240 - e) / 60 : t) * 255;
}
const Ms = (e) => () => e;
function eu(e, t) {
  return function(r) {
    return e + r * t;
  };
}
function cx(e, t, r) {
  return e = Math.pow(e, r), t = Math.pow(t, r) - e, r = 1 / r, function(i) {
    return Math.pow(e + i * t, r);
  };
}
function TT(e, t) {
  var r = t - e;
  return r ? eu(e, r > 180 || r < -180 ? r - 360 * Math.round(r / 360) : r) : Ms(isNaN(e) ? t : e);
}
function hx(e) {
  return (e = +e) == 1 ? ru : function(t, r) {
    return r - t ? cx(t, r, e) : Ms(isNaN(t) ? r : t);
  };
}
function ru(e, t) {
  var r = t - e;
  return r ? eu(e, r) : Ms(isNaN(e) ? t : e);
}
const Ao = function e(t) {
  var r = hx(t);
  function i(a, n) {
    var o = r((a = Rn(a)).r, (n = Rn(n)).r), s = r(a.g, n.g), l = r(a.b, n.b), c = ru(a.opacity, n.opacity);
    return function(h) {
      return a.r = o(h), a.g = s(h), a.b = l(h), a.opacity = c(h), a + "";
    };
  }
  return i.gamma = e, i;
}(1);
function Ce(e, t) {
  return e = +e, t = +t, function(r) {
    return e * (1 - r) + t * r;
  };
}
var Pn = /[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g, tn = new RegExp(Pn.source, "g");
function ux(e) {
  return function() {
    return e;
  };
}
function fx(e) {
  return function(t) {
    return e(t) + "";
  };
}
function dx(e, t) {
  var r = Pn.lastIndex = tn.lastIndex = 0, i, a, n, o = -1, s = [], l = [];
  for (e = e + "", t = t + ""; (i = Pn.exec(e)) && (a = tn.exec(t)); )
    (n = a.index) > r && (n = t.slice(r, n), s[o] ? s[o] += n : s[++o] = n), (i = i[0]) === (a = a[0]) ? s[o] ? s[o] += a : s[++o] = a : (s[++o] = null, l.push({ i: o, x: Ce(i, a) })), r = tn.lastIndex;
  return r < t.length && (n = t.slice(r), s[o] ? s[o] += n : s[++o] = n), s.length < 2 ? l[0] ? fx(l[0].x) : ux(t) : (t = l.length, function(c) {
    for (var h = 0, u; h < t; ++h) s[(u = l[h]).i] = u.x(c);
    return s.join("");
  });
}
var Fo = 180 / Math.PI, In = {
  translateX: 0,
  translateY: 0,
  rotate: 0,
  skewX: 0,
  scaleX: 1,
  scaleY: 1
};
function iu(e, t, r, i, a, n) {
  var o, s, l;
  return (o = Math.sqrt(e * e + t * t)) && (e /= o, t /= o), (l = e * r + t * i) && (r -= e * l, i -= t * l), (s = Math.sqrt(r * r + i * i)) && (r /= s, i /= s, l /= s), e * i < t * r && (e = -e, t = -t, l = -l, o = -o), {
    translateX: a,
    translateY: n,
    rotate: Math.atan2(t, e) * Fo,
    skewX: Math.atan(l) * Fo,
    scaleX: o,
    scaleY: s
  };
}
var wi;
function px(e) {
  const t = new (typeof DOMMatrix == "function" ? DOMMatrix : WebKitCSSMatrix)(e + "");
  return t.isIdentity ? In : iu(t.a, t.b, t.c, t.d, t.e, t.f);
}
function gx(e) {
  return e == null || (wi || (wi = document.createElementNS("http://www.w3.org/2000/svg", "g")), wi.setAttribute("transform", e), !(e = wi.transform.baseVal.consolidate())) ? In : (e = e.matrix, iu(e.a, e.b, e.c, e.d, e.e, e.f));
}
function au(e, t, r, i) {
  function a(c) {
    return c.length ? c.pop() + " " : "";
  }
  function n(c, h, u, f, d, g) {
    if (c !== u || h !== f) {
      var m = d.push("translate(", null, t, null, r);
      g.push({ i: m - 4, x: Ce(c, u) }, { i: m - 2, x: Ce(h, f) });
    } else (u || f) && d.push("translate(" + u + t + f + r);
  }
  function o(c, h, u, f) {
    c !== h ? (c - h > 180 ? h += 360 : h - c > 180 && (c += 360), f.push({ i: u.push(a(u) + "rotate(", null, i) - 2, x: Ce(c, h) })) : h && u.push(a(u) + "rotate(" + h + i);
  }
  function s(c, h, u, f) {
    c !== h ? f.push({ i: u.push(a(u) + "skewX(", null, i) - 2, x: Ce(c, h) }) : h && u.push(a(u) + "skewX(" + h + i);
  }
  function l(c, h, u, f, d, g) {
    if (c !== u || h !== f) {
      var m = d.push(a(d) + "scale(", null, ",", null, ")");
      g.push({ i: m - 4, x: Ce(c, u) }, { i: m - 2, x: Ce(h, f) });
    } else (u !== 1 || f !== 1) && d.push(a(d) + "scale(" + u + "," + f + ")");
  }
  return function(c, h) {
    var u = [], f = [];
    return c = e(c), h = e(h), n(c.translateX, c.translateY, h.translateX, h.translateY, u, f), o(c.rotate, h.rotate, u, f), s(c.skewX, h.skewX, u, f), l(c.scaleX, c.scaleY, h.scaleX, h.scaleY, u, f), c = h = null, function(d) {
      for (var g = -1, m = f.length, y; ++g < m; ) u[(y = f[g]).i] = y.x(d);
      return u.join("");
    };
  };
}
var mx = au(px, "px, ", "px)", "deg)"), yx = au(gx, ", ", ")", ")"), kr = 0, zr = 0, Ar = 0, nu = 1e3, ra, qr, ia = 0, He = 0, Fa = 0, ai = typeof performance == "object" && performance.now ? performance : Date, su = typeof window == "object" && window.requestAnimationFrame ? window.requestAnimationFrame.bind(window) : function(e) {
  setTimeout(e, 17);
};
function $s() {
  return He || (su(xx), He = ai.now() + Fa);
}
function xx() {
  He = 0;
}
function aa() {
  this._call = this._time = this._next = null;
}
aa.prototype = ou.prototype = {
  constructor: aa,
  restart: function(e, t, r) {
    if (typeof e != "function") throw new TypeError("callback is not a function");
    r = (r == null ? $s() : +r) + (t == null ? 0 : +t), !this._next && qr !== this && (qr ? qr._next = this : ra = this, qr = this), this._call = e, this._time = r, Nn();
  },
  stop: function() {
    this._call && (this._call = null, this._time = 1 / 0, Nn());
  }
};
function ou(e, t, r) {
  var i = new aa();
  return i.restart(e, t, r), i;
}
function bx() {
  $s(), ++kr;
  for (var e = ra, t; e; )
    (t = He - e._time) >= 0 && e._call.call(void 0, t), e = e._next;
  --kr;
}
function Eo() {
  He = (ia = ai.now()) + Fa, kr = zr = 0;
  try {
    bx();
  } finally {
    kr = 0, kx(), He = 0;
  }
}
function Cx() {
  var e = ai.now(), t = e - ia;
  t > nu && (Fa -= t, ia = e);
}
function kx() {
  for (var e, t = ra, r, i = 1 / 0; t; )
    t._call ? (i > t._time && (i = t._time), e = t, t = t._next) : (r = t._next, t._next = null, t = e ? e._next = r : ra = r);
  qr = e, Nn(i);
}
function Nn(e) {
  if (!kr) {
    zr && (zr = clearTimeout(zr));
    var t = e - He;
    t > 24 ? (e < 1 / 0 && (zr = setTimeout(Eo, e - ai.now() - Fa)), Ar && (Ar = clearInterval(Ar))) : (Ar || (ia = ai.now(), Ar = setInterval(Cx, nu)), kr = 1, su(Eo));
  }
}
function Oo(e, t, r) {
  var i = new aa();
  return t = t == null ? 0 : +t, i.restart((a) => {
    i.stop(), e(a + t);
  }, t, r), i;
}
var wx = zh("start", "end", "cancel", "interrupt"), _x = [], lu = 0, Do = 1, zn = 2, Oi = 3, Ro = 4, qn = 5, Di = 6;
function Ea(e, t, r, i, a, n) {
  var o = e.__transition;
  if (!o) e.__transition = {};
  else if (r in o) return;
  vx(e, r, {
    name: t,
    index: i,
    // For context during callback.
    group: a,
    // For context during callback.
    on: wx,
    tween: _x,
    time: n.time,
    delay: n.delay,
    duration: n.duration,
    ease: n.ease,
    timer: null,
    state: lu
  });
}
function As(e, t) {
  var r = Xt(e, t);
  if (r.state > lu) throw new Error("too late; already scheduled");
  return r;
}
function ie(e, t) {
  var r = Xt(e, t);
  if (r.state > Oi) throw new Error("too late; already running");
  return r;
}
function Xt(e, t) {
  var r = e.__transition;
  if (!r || !(r = r[t])) throw new Error("transition not found");
  return r;
}
function vx(e, t, r) {
  var i = e.__transition, a;
  i[t] = r, r.timer = ou(n, 0, r.time);
  function n(c) {
    r.state = Do, r.timer.restart(o, r.delay, r.time), r.delay <= c && o(c - r.delay);
  }
  function o(c) {
    var h, u, f, d;
    if (r.state !== Do) return l();
    for (h in i)
      if (d = i[h], d.name === r.name) {
        if (d.state === Oi) return Oo(o);
        d.state === Ro ? (d.state = Di, d.timer.stop(), d.on.call("interrupt", e, e.__data__, d.index, d.group), delete i[h]) : +h < t && (d.state = Di, d.timer.stop(), d.on.call("cancel", e, e.__data__, d.index, d.group), delete i[h]);
      }
    if (Oo(function() {
      r.state === Oi && (r.state = Ro, r.timer.restart(s, r.delay, r.time), s(c));
    }), r.state = zn, r.on.call("start", e, e.__data__, r.index, r.group), r.state === zn) {
      for (r.state = Oi, a = new Array(f = r.tween.length), h = 0, u = -1; h < f; ++h)
        (d = r.tween[h].value.call(e, e.__data__, r.index, r.group)) && (a[++u] = d);
      a.length = u + 1;
    }
  }
  function s(c) {
    for (var h = c < r.duration ? r.ease.call(null, c / r.duration) : (r.timer.restart(l), r.state = qn, 1), u = -1, f = a.length; ++u < f; )
      a[u].call(e, h);
    r.state === qn && (r.on.call("end", e, e.__data__, r.index, r.group), l());
  }
  function l() {
    r.state = Di, r.timer.stop(), delete i[t];
    for (var c in i) return;
    delete e.__transition;
  }
}
function Sx(e, t) {
  var r = e.__transition, i, a, n = !0, o;
  if (r) {
    t = t == null ? null : t + "";
    for (o in r) {
      if ((i = r[o]).name !== t) {
        n = !1;
        continue;
      }
      a = i.state > zn && i.state < qn, i.state = Di, i.timer.stop(), i.on.call(a ? "interrupt" : "cancel", e, e.__data__, i.index, i.group), delete r[o];
    }
    n && delete e.__transition;
  }
}
function Tx(e) {
  return this.each(function() {
    Sx(this, e);
  });
}
function Bx(e, t) {
  var r, i;
  return function() {
    var a = ie(this, e), n = a.tween;
    if (n !== r) {
      i = r = n;
      for (var o = 0, s = i.length; o < s; ++o)
        if (i[o].name === t) {
          i = i.slice(), i.splice(o, 1);
          break;
        }
    }
    a.tween = i;
  };
}
function Lx(e, t, r) {
  var i, a;
  if (typeof r != "function") throw new Error();
  return function() {
    var n = ie(this, e), o = n.tween;
    if (o !== i) {
      a = (i = o).slice();
      for (var s = { name: t, value: r }, l = 0, c = a.length; l < c; ++l)
        if (a[l].name === t) {
          a[l] = s;
          break;
        }
      l === c && a.push(s);
    }
    n.tween = a;
  };
}
function Mx(e, t) {
  var r = this._id;
  if (e += "", arguments.length < 2) {
    for (var i = Xt(this.node(), r).tween, a = 0, n = i.length, o; a < n; ++a)
      if ((o = i[a]).name === e)
        return o.value;
    return null;
  }
  return this.each((t == null ? Bx : Lx)(r, e, t));
}
function Fs(e, t, r) {
  var i = e._id;
  return e.each(function() {
    var a = ie(this, i);
    (a.value || (a.value = {}))[t] = r.apply(this, arguments);
  }), function(a) {
    return Xt(a, i).value[t];
  };
}
function cu(e, t) {
  var r;
  return (typeof t == "number" ? Ce : t instanceof ii ? Ao : (r = ii(t)) ? (t = r, Ao) : dx)(e, t);
}
function $x(e) {
  return function() {
    this.removeAttribute(e);
  };
}
function Ax(e) {
  return function() {
    this.removeAttributeNS(e.space, e.local);
  };
}
function Fx(e, t, r) {
  var i, a = r + "", n;
  return function() {
    var o = this.getAttribute(e);
    return o === a ? null : o === i ? n : n = t(i = o, r);
  };
}
function Ex(e, t, r) {
  var i, a = r + "", n;
  return function() {
    var o = this.getAttributeNS(e.space, e.local);
    return o === a ? null : o === i ? n : n = t(i = o, r);
  };
}
function Ox(e, t, r) {
  var i, a, n;
  return function() {
    var o, s = r(this), l;
    return s == null ? void this.removeAttribute(e) : (o = this.getAttribute(e), l = s + "", o === l ? null : o === i && l === a ? n : (a = l, n = t(i = o, s)));
  };
}
function Dx(e, t, r) {
  var i, a, n;
  return function() {
    var o, s = r(this), l;
    return s == null ? void this.removeAttributeNS(e.space, e.local) : (o = this.getAttributeNS(e.space, e.local), l = s + "", o === l ? null : o === i && l === a ? n : (a = l, n = t(i = o, s)));
  };
}
function Rx(e, t) {
  var r = Aa(e), i = r === "transform" ? yx : cu;
  return this.attrTween(e, typeof t == "function" ? (r.local ? Dx : Ox)(r, i, Fs(this, "attr." + e, t)) : t == null ? (r.local ? Ax : $x)(r) : (r.local ? Ex : Fx)(r, i, t));
}
function Px(e, t) {
  return function(r) {
    this.setAttribute(e, t.call(this, r));
  };
}
function Ix(e, t) {
  return function(r) {
    this.setAttributeNS(e.space, e.local, t.call(this, r));
  };
}
function Nx(e, t) {
  var r, i;
  function a() {
    var n = t.apply(this, arguments);
    return n !== i && (r = (i = n) && Ix(e, n)), r;
  }
  return a._value = t, a;
}
function zx(e, t) {
  var r, i;
  function a() {
    var n = t.apply(this, arguments);
    return n !== i && (r = (i = n) && Px(e, n)), r;
  }
  return a._value = t, a;
}
function qx(e, t) {
  var r = "attr." + e;
  if (arguments.length < 2) return (r = this.tween(r)) && r._value;
  if (t == null) return this.tween(r, null);
  if (typeof t != "function") throw new Error();
  var i = Aa(e);
  return this.tween(r, (i.local ? Nx : zx)(i, t));
}
function Wx(e, t) {
  return function() {
    As(this, e).delay = +t.apply(this, arguments);
  };
}
function Hx(e, t) {
  return t = +t, function() {
    As(this, e).delay = t;
  };
}
function jx(e) {
  var t = this._id;
  return arguments.length ? this.each((typeof e == "function" ? Wx : Hx)(t, e)) : Xt(this.node(), t).delay;
}
function Yx(e, t) {
  return function() {
    ie(this, e).duration = +t.apply(this, arguments);
  };
}
function Gx(e, t) {
  return t = +t, function() {
    ie(this, e).duration = t;
  };
}
function Ux(e) {
  var t = this._id;
  return arguments.length ? this.each((typeof e == "function" ? Yx : Gx)(t, e)) : Xt(this.node(), t).duration;
}
function Xx(e, t) {
  if (typeof t != "function") throw new Error();
  return function() {
    ie(this, e).ease = t;
  };
}
function Vx(e) {
  var t = this._id;
  return arguments.length ? this.each(Xx(t, e)) : Xt(this.node(), t).ease;
}
function Zx(e, t) {
  return function() {
    var r = t.apply(this, arguments);
    if (typeof r != "function") throw new Error();
    ie(this, e).ease = r;
  };
}
function Kx(e) {
  if (typeof e != "function") throw new Error();
  return this.each(Zx(this._id, e));
}
function Qx(e) {
  typeof e != "function" && (e = Hh(e));
  for (var t = this._groups, r = t.length, i = new Array(r), a = 0; a < r; ++a)
    for (var n = t[a], o = n.length, s = i[a] = [], l, c = 0; c < o; ++c)
      (l = n[c]) && e.call(l, l.__data__, c, n) && s.push(l);
  return new de(i, this._parents, this._name, this._id);
}
function Jx(e) {
  if (e._id !== this._id) throw new Error();
  for (var t = this._groups, r = e._groups, i = t.length, a = r.length, n = Math.min(i, a), o = new Array(i), s = 0; s < n; ++s)
    for (var l = t[s], c = r[s], h = l.length, u = o[s] = new Array(h), f, d = 0; d < h; ++d)
      (f = l[d] || c[d]) && (u[d] = f);
  for (; s < i; ++s)
    o[s] = t[s];
  return new de(o, this._parents, this._name, this._id);
}
function tb(e) {
  return (e + "").trim().split(/^|\s+/).every(function(t) {
    var r = t.indexOf(".");
    return r >= 0 && (t = t.slice(0, r)), !t || t === "start";
  });
}
function eb(e, t, r) {
  var i, a, n = tb(t) ? As : ie;
  return function() {
    var o = n(this, e), s = o.on;
    s !== i && (a = (i = s).copy()).on(t, r), o.on = a;
  };
}
function rb(e, t) {
  var r = this._id;
  return arguments.length < 2 ? Xt(this.node(), r).on.on(e) : this.each(eb(r, e, t));
}
function ib(e) {
  return function() {
    var t = this.parentNode;
    for (var r in this.__transition) if (+r !== e) return;
    t && t.removeChild(this);
  };
}
function ab() {
  return this.on("end.remove", ib(this._id));
}
function nb(e) {
  var t = this._name, r = this._id;
  typeof e != "function" && (e = Ts(e));
  for (var i = this._groups, a = i.length, n = new Array(a), o = 0; o < a; ++o)
    for (var s = i[o], l = s.length, c = n[o] = new Array(l), h, u, f = 0; f < l; ++f)
      (h = s[f]) && (u = e.call(h, h.__data__, f, s)) && ("__data__" in h && (u.__data__ = h.__data__), c[f] = u, Ea(c[f], t, r, f, c, Xt(h, r)));
  return new de(n, this._parents, t, r);
}
function sb(e) {
  var t = this._name, r = this._id;
  typeof e != "function" && (e = Wh(e));
  for (var i = this._groups, a = i.length, n = [], o = [], s = 0; s < a; ++s)
    for (var l = i[s], c = l.length, h, u = 0; u < c; ++u)
      if (h = l[u]) {
        for (var f = e.call(h, h.__data__, u, l), d, g = Xt(h, r), m = 0, y = f.length; m < y; ++m)
          (d = f[m]) && Ea(d, t, r, m, f, g);
        n.push(f), o.push(h);
      }
  return new de(n, o, t, r);
}
var ob = fi.prototype.constructor;
function lb() {
  return new ob(this._groups, this._parents);
}
function cb(e, t) {
  var r, i, a;
  return function() {
    var n = Cr(this, e), o = (this.style.removeProperty(e), Cr(this, e));
    return n === o ? null : n === r && o === i ? a : a = t(r = n, i = o);
  };
}
function hu(e) {
  return function() {
    this.style.removeProperty(e);
  };
}
function hb(e, t, r) {
  var i, a = r + "", n;
  return function() {
    var o = Cr(this, e);
    return o === a ? null : o === i ? n : n = t(i = o, r);
  };
}
function ub(e, t, r) {
  var i, a, n;
  return function() {
    var o = Cr(this, e), s = r(this), l = s + "";
    return s == null && (l = s = (this.style.removeProperty(e), Cr(this, e))), o === l ? null : o === i && l === a ? n : (a = l, n = t(i = o, s));
  };
}
function fb(e, t) {
  var r, i, a, n = "style." + t, o = "end." + n, s;
  return function() {
    var l = ie(this, e), c = l.on, h = l.value[n] == null ? s || (s = hu(t)) : void 0;
    (c !== r || a !== h) && (i = (r = c).copy()).on(o, a = h), l.on = i;
  };
}
function db(e, t, r) {
  var i = (e += "") == "transform" ? mx : cu;
  return t == null ? this.styleTween(e, cb(e, i)).on("end.style." + e, hu(e)) : typeof t == "function" ? this.styleTween(e, ub(e, i, Fs(this, "style." + e, t))).each(fb(this._id, e)) : this.styleTween(e, hb(e, i, t), r).on("end.style." + e, null);
}
function pb(e, t, r) {
  return function(i) {
    this.style.setProperty(e, t.call(this, i), r);
  };
}
function gb(e, t, r) {
  var i, a;
  function n() {
    var o = t.apply(this, arguments);
    return o !== a && (i = (a = o) && pb(e, o, r)), i;
  }
  return n._value = t, n;
}
function mb(e, t, r) {
  var i = "style." + (e += "");
  if (arguments.length < 2) return (i = this.tween(i)) && i._value;
  if (t == null) return this.tween(i, null);
  if (typeof t != "function") throw new Error();
  return this.tween(i, gb(e, t, r ?? ""));
}
function yb(e) {
  return function() {
    this.textContent = e;
  };
}
function xb(e) {
  return function() {
    var t = e(this);
    this.textContent = t ?? "";
  };
}
function bb(e) {
  return this.tween("text", typeof e == "function" ? xb(Fs(this, "text", e)) : yb(e == null ? "" : e + ""));
}
function Cb(e) {
  return function(t) {
    this.textContent = e.call(this, t);
  };
}
function kb(e) {
  var t, r;
  function i() {
    var a = e.apply(this, arguments);
    return a !== r && (t = (r = a) && Cb(a)), t;
  }
  return i._value = e, i;
}
function wb(e) {
  var t = "text";
  if (arguments.length < 1) return (t = this.tween(t)) && t._value;
  if (e == null) return this.tween(t, null);
  if (typeof e != "function") throw new Error();
  return this.tween(t, kb(e));
}
function _b() {
  for (var e = this._name, t = this._id, r = uu(), i = this._groups, a = i.length, n = 0; n < a; ++n)
    for (var o = i[n], s = o.length, l, c = 0; c < s; ++c)
      if (l = o[c]) {
        var h = Xt(l, t);
        Ea(l, e, r, c, o, {
          time: h.time + h.delay + h.duration,
          delay: 0,
          duration: h.duration,
          ease: h.ease
        });
      }
  return new de(i, this._parents, e, r);
}
function vb() {
  var e, t, r = this, i = r._id, a = r.size();
  return new Promise(function(n, o) {
    var s = { value: o }, l = { value: function() {
      --a === 0 && n();
    } };
    r.each(function() {
      var c = ie(this, i), h = c.on;
      h !== e && (t = (e = h).copy(), t._.cancel.push(s), t._.interrupt.push(s), t._.end.push(l)), c.on = t;
    }), a === 0 && n();
  });
}
var Sb = 0;
function de(e, t, r, i) {
  this._groups = e, this._parents = t, this._name = r, this._id = i;
}
function uu() {
  return ++Sb;
}
var se = fi.prototype;
de.prototype = {
  constructor: de,
  select: nb,
  selectAll: sb,
  selectChild: se.selectChild,
  selectChildren: se.selectChildren,
  filter: Qx,
  merge: Jx,
  selection: lb,
  transition: _b,
  call: se.call,
  nodes: se.nodes,
  node: se.node,
  size: se.size,
  empty: se.empty,
  each: se.each,
  on: rb,
  attr: Rx,
  attrTween: qx,
  style: db,
  styleTween: mb,
  text: bb,
  textTween: wb,
  remove: ab,
  tween: Mx,
  delay: jx,
  duration: Ux,
  ease: Vx,
  easeVarying: Kx,
  end: vb,
  [Symbol.iterator]: se[Symbol.iterator]
};
function Tb(e) {
  return ((e *= 2) <= 1 ? e * e * e : (e -= 2) * e * e + 2) / 2;
}
var Bb = {
  time: null,
  // Set on use.
  delay: 0,
  duration: 250,
  ease: Tb
};
function Lb(e, t) {
  for (var r; !(r = e.__transition) || !(r = r[t]); )
    if (!(e = e.parentNode))
      throw new Error(`transition ${t} not found`);
  return r;
}
function Mb(e) {
  var t, r;
  e instanceof de ? (t = e._id, e = e._name) : (t = uu(), (r = Bb).time = $s(), e = e == null ? null : e + "");
  for (var i = this._groups, a = i.length, n = 0; n < a; ++n)
    for (var o = i[n], s = o.length, l, c = 0; c < s; ++c)
      (l = o[c]) && Ea(l, e, t, c, o, r || Lb(l, t));
  return new de(i, this._parents, e, t);
}
fi.prototype.interrupt = Tx;
fi.prototype.transition = Mb;
const Wn = Math.PI, Hn = 2 * Wn, $e = 1e-6, $b = Hn - $e;
function fu(e) {
  this._ += e[0];
  for (let t = 1, r = e.length; t < r; ++t)
    this._ += arguments[t] + e[t];
}
function Ab(e) {
  let t = Math.floor(e);
  if (!(t >= 0)) throw new Error(`invalid digits: ${e}`);
  if (t > 15) return fu;
  const r = 10 ** t;
  return function(i) {
    this._ += i[0];
    for (let a = 1, n = i.length; a < n; ++a)
      this._ += Math.round(arguments[a] * r) / r + i[a];
  };
}
class Fb {
  constructor(t) {
    this._x0 = this._y0 = // start of current subpath
    this._x1 = this._y1 = null, this._ = "", this._append = t == null ? fu : Ab(t);
  }
  moveTo(t, r) {
    this._append`M${this._x0 = this._x1 = +t},${this._y0 = this._y1 = +r}`;
  }
  closePath() {
    this._x1 !== null && (this._x1 = this._x0, this._y1 = this._y0, this._append`Z`);
  }
  lineTo(t, r) {
    this._append`L${this._x1 = +t},${this._y1 = +r}`;
  }
  quadraticCurveTo(t, r, i, a) {
    this._append`Q${+t},${+r},${this._x1 = +i},${this._y1 = +a}`;
  }
  bezierCurveTo(t, r, i, a, n, o) {
    this._append`C${+t},${+r},${+i},${+a},${this._x1 = +n},${this._y1 = +o}`;
  }
  arcTo(t, r, i, a, n) {
    if (t = +t, r = +r, i = +i, a = +a, n = +n, n < 0) throw new Error(`negative radius: ${n}`);
    let o = this._x1, s = this._y1, l = i - t, c = a - r, h = o - t, u = s - r, f = h * h + u * u;
    if (this._x1 === null)
      this._append`M${this._x1 = t},${this._y1 = r}`;
    else if (f > $e) if (!(Math.abs(u * l - c * h) > $e) || !n)
      this._append`L${this._x1 = t},${this._y1 = r}`;
    else {
      let d = i - o, g = a - s, m = l * l + c * c, y = d * d + g * g, x = Math.sqrt(m), b = Math.sqrt(f), k = n * Math.tan((Wn - Math.acos((m + f - y) / (2 * x * b))) / 2), S = k / b, w = k / x;
      Math.abs(S - 1) > $e && this._append`L${t + S * h},${r + S * u}`, this._append`A${n},${n},0,0,${+(u * d > h * g)},${this._x1 = t + w * l},${this._y1 = r + w * c}`;
    }
  }
  arc(t, r, i, a, n, o) {
    if (t = +t, r = +r, i = +i, o = !!o, i < 0) throw new Error(`negative radius: ${i}`);
    let s = i * Math.cos(a), l = i * Math.sin(a), c = t + s, h = r + l, u = 1 ^ o, f = o ? a - n : n - a;
    this._x1 === null ? this._append`M${c},${h}` : (Math.abs(this._x1 - c) > $e || Math.abs(this._y1 - h) > $e) && this._append`L${c},${h}`, i && (f < 0 && (f = f % Hn + Hn), f > $b ? this._append`A${i},${i},0,1,${u},${t - s},${r - l}A${i},${i},0,1,${u},${this._x1 = c},${this._y1 = h}` : f > $e && this._append`A${i},${i},0,${+(f >= Wn)},${u},${this._x1 = t + i * Math.cos(n)},${this._y1 = r + i * Math.sin(n)}`);
  }
  rect(t, r, i, a) {
    this._append`M${this._x0 = this._x1 = +t},${this._y0 = this._y1 = +r}h${i = +i}v${+a}h${-i}Z`;
  }
  toString() {
    return this._;
  }
}
function Qe(e) {
  return function() {
    return e;
  };
}
const BT = Math.abs, LT = Math.atan2, MT = Math.cos, $T = Math.max, AT = Math.min, FT = Math.sin, ET = Math.sqrt, Po = 1e-12, Es = Math.PI, Io = Es / 2, OT = 2 * Es;
function DT(e) {
  return e > 1 ? 0 : e < -1 ? Es : Math.acos(e);
}
function RT(e) {
  return e >= 1 ? Io : e <= -1 ? -Io : Math.asin(e);
}
function Eb(e) {
  let t = 3;
  return e.digits = function(r) {
    if (!arguments.length) return t;
    if (r == null)
      t = null;
    else {
      const i = Math.floor(r);
      if (!(i >= 0)) throw new RangeError(`invalid digits: ${r}`);
      t = i;
    }
    return e;
  }, () => new Fb(t);
}
function Ob(e) {
  return typeof e == "object" && "length" in e ? e : Array.from(e);
}
function du(e) {
  this._context = e;
}
du.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._point = 0;
  },
  lineEnd: function() {
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
        break;
      case 1:
        this._point = 2;
      default:
        this._context.lineTo(e, t);
        break;
    }
  }
};
function na(e) {
  return new du(e);
}
function Db(e) {
  return e[0];
}
function Rb(e) {
  return e[1];
}
function Pb(e, t) {
  var r = Qe(!0), i = null, a = na, n = null, o = Eb(s);
  e = typeof e == "function" ? e : e === void 0 ? Db : Qe(e), t = typeof t == "function" ? t : t === void 0 ? Rb : Qe(t);
  function s(l) {
    var c, h = (l = Ob(l)).length, u, f = !1, d;
    for (i == null && (n = a(d = o())), c = 0; c <= h; ++c)
      !(c < h && r(u = l[c], c, l)) === f && ((f = !f) ? n.lineStart() : n.lineEnd()), f && n.point(+e(u, c, l), +t(u, c, l));
    if (d) return n = null, d + "" || null;
  }
  return s.x = function(l) {
    return arguments.length ? (e = typeof l == "function" ? l : Qe(+l), s) : e;
  }, s.y = function(l) {
    return arguments.length ? (t = typeof l == "function" ? l : Qe(+l), s) : t;
  }, s.defined = function(l) {
    return arguments.length ? (r = typeof l == "function" ? l : Qe(!!l), s) : r;
  }, s.curve = function(l) {
    return arguments.length ? (a = l, i != null && (n = a(i)), s) : a;
  }, s.context = function(l) {
    return arguments.length ? (l == null ? i = n = null : n = a(i = l), s) : i;
  }, s;
}
class pu {
  constructor(t, r) {
    this._context = t, this._x = r;
  }
  areaStart() {
    this._line = 0;
  }
  areaEnd() {
    this._line = NaN;
  }
  lineStart() {
    this._point = 0;
  }
  lineEnd() {
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  }
  point(t, r) {
    switch (t = +t, r = +r, this._point) {
      case 0: {
        this._point = 1, this._line ? this._context.lineTo(t, r) : this._context.moveTo(t, r);
        break;
      }
      case 1:
        this._point = 2;
      default: {
        this._x ? this._context.bezierCurveTo(this._x0 = (this._x0 + t) / 2, this._y0, this._x0, r, t, r) : this._context.bezierCurveTo(this._x0, this._y0 = (this._y0 + r) / 2, t, this._y0, t, r);
        break;
      }
    }
    this._x0 = t, this._y0 = r;
  }
}
function gu(e) {
  return new pu(e, !0);
}
function mu(e) {
  return new pu(e, !1);
}
function ve() {
}
function sa(e, t, r) {
  e._context.bezierCurveTo(
    (2 * e._x0 + e._x1) / 3,
    (2 * e._y0 + e._y1) / 3,
    (e._x0 + 2 * e._x1) / 3,
    (e._y0 + 2 * e._y1) / 3,
    (e._x0 + 4 * e._x1 + t) / 6,
    (e._y0 + 4 * e._y1 + r) / 6
  );
}
function Oa(e) {
  this._context = e;
}
Oa.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._y0 = this._y1 = NaN, this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 3:
        sa(this, this._x1, this._y1);
      case 2:
        this._context.lineTo(this._x1, this._y1);
        break;
    }
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
        break;
      case 1:
        this._point = 2;
        break;
      case 2:
        this._point = 3, this._context.lineTo((5 * this._x0 + this._x1) / 6, (5 * this._y0 + this._y1) / 6);
      default:
        sa(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = e, this._y0 = this._y1, this._y1 = t;
  }
};
function Ri(e) {
  return new Oa(e);
}
function yu(e) {
  this._context = e;
}
yu.prototype = {
  areaStart: ve,
  areaEnd: ve,
  lineStart: function() {
    this._x0 = this._x1 = this._x2 = this._x3 = this._x4 = this._y0 = this._y1 = this._y2 = this._y3 = this._y4 = NaN, this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 1: {
        this._context.moveTo(this._x2, this._y2), this._context.closePath();
        break;
      }
      case 2: {
        this._context.moveTo((this._x2 + 2 * this._x3) / 3, (this._y2 + 2 * this._y3) / 3), this._context.lineTo((this._x3 + 2 * this._x2) / 3, (this._y3 + 2 * this._y2) / 3), this._context.closePath();
        break;
      }
      case 3: {
        this.point(this._x2, this._y2), this.point(this._x3, this._y3), this.point(this._x4, this._y4);
        break;
      }
    }
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._x2 = e, this._y2 = t;
        break;
      case 1:
        this._point = 2, this._x3 = e, this._y3 = t;
        break;
      case 2:
        this._point = 3, this._x4 = e, this._y4 = t, this._context.moveTo((this._x0 + 4 * this._x1 + e) / 6, (this._y0 + 4 * this._y1 + t) / 6);
        break;
      default:
        sa(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = e, this._y0 = this._y1, this._y1 = t;
  }
};
function Ib(e) {
  return new yu(e);
}
function xu(e) {
  this._context = e;
}
xu.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._y0 = this._y1 = NaN, this._point = 0;
  },
  lineEnd: function() {
    (this._line || this._line !== 0 && this._point === 3) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1;
        break;
      case 1:
        this._point = 2;
        break;
      case 2:
        this._point = 3;
        var r = (this._x0 + 4 * this._x1 + e) / 6, i = (this._y0 + 4 * this._y1 + t) / 6;
        this._line ? this._context.lineTo(r, i) : this._context.moveTo(r, i);
        break;
      case 3:
        this._point = 4;
      default:
        sa(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = e, this._y0 = this._y1, this._y1 = t;
  }
};
function Nb(e) {
  return new xu(e);
}
function bu(e, t) {
  this._basis = new Oa(e), this._beta = t;
}
bu.prototype = {
  lineStart: function() {
    this._x = [], this._y = [], this._basis.lineStart();
  },
  lineEnd: function() {
    var e = this._x, t = this._y, r = e.length - 1;
    if (r > 0)
      for (var i = e[0], a = t[0], n = e[r] - i, o = t[r] - a, s = -1, l; ++s <= r; )
        l = s / r, this._basis.point(
          this._beta * e[s] + (1 - this._beta) * (i + l * n),
          this._beta * t[s] + (1 - this._beta) * (a + l * o)
        );
    this._x = this._y = null, this._basis.lineEnd();
  },
  point: function(e, t) {
    this._x.push(+e), this._y.push(+t);
  }
};
const zb = function e(t) {
  function r(i) {
    return t === 1 ? new Oa(i) : new bu(i, t);
  }
  return r.beta = function(i) {
    return e(+i);
  }, r;
}(0.85);
function oa(e, t, r) {
  e._context.bezierCurveTo(
    e._x1 + e._k * (e._x2 - e._x0),
    e._y1 + e._k * (e._y2 - e._y0),
    e._x2 + e._k * (e._x1 - t),
    e._y2 + e._k * (e._y1 - r),
    e._x2,
    e._y2
  );
}
function Os(e, t) {
  this._context = e, this._k = (1 - t) / 6;
}
Os.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._x2 = this._y0 = this._y1 = this._y2 = NaN, this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 2:
        this._context.lineTo(this._x2, this._y2);
        break;
      case 3:
        oa(this, this._x1, this._y1);
        break;
    }
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
        break;
      case 1:
        this._point = 2, this._x1 = e, this._y1 = t;
        break;
      case 2:
        this._point = 3;
      default:
        oa(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = this._x2, this._x2 = e, this._y0 = this._y1, this._y1 = this._y2, this._y2 = t;
  }
};
const Cu = function e(t) {
  function r(i) {
    return new Os(i, t);
  }
  return r.tension = function(i) {
    return e(+i);
  }, r;
}(0);
function Ds(e, t) {
  this._context = e, this._k = (1 - t) / 6;
}
Ds.prototype = {
  areaStart: ve,
  areaEnd: ve,
  lineStart: function() {
    this._x0 = this._x1 = this._x2 = this._x3 = this._x4 = this._x5 = this._y0 = this._y1 = this._y2 = this._y3 = this._y4 = this._y5 = NaN, this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 1: {
        this._context.moveTo(this._x3, this._y3), this._context.closePath();
        break;
      }
      case 2: {
        this._context.lineTo(this._x3, this._y3), this._context.closePath();
        break;
      }
      case 3: {
        this.point(this._x3, this._y3), this.point(this._x4, this._y4), this.point(this._x5, this._y5);
        break;
      }
    }
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._x3 = e, this._y3 = t;
        break;
      case 1:
        this._point = 2, this._context.moveTo(this._x4 = e, this._y4 = t);
        break;
      case 2:
        this._point = 3, this._x5 = e, this._y5 = t;
        break;
      default:
        oa(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = this._x2, this._x2 = e, this._y0 = this._y1, this._y1 = this._y2, this._y2 = t;
  }
};
const qb = function e(t) {
  function r(i) {
    return new Ds(i, t);
  }
  return r.tension = function(i) {
    return e(+i);
  }, r;
}(0);
function Rs(e, t) {
  this._context = e, this._k = (1 - t) / 6;
}
Rs.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._x2 = this._y0 = this._y1 = this._y2 = NaN, this._point = 0;
  },
  lineEnd: function() {
    (this._line || this._line !== 0 && this._point === 3) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1;
        break;
      case 1:
        this._point = 2;
        break;
      case 2:
        this._point = 3, this._line ? this._context.lineTo(this._x2, this._y2) : this._context.moveTo(this._x2, this._y2);
        break;
      case 3:
        this._point = 4;
      default:
        oa(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = this._x2, this._x2 = e, this._y0 = this._y1, this._y1 = this._y2, this._y2 = t;
  }
};
const Wb = function e(t) {
  function r(i) {
    return new Rs(i, t);
  }
  return r.tension = function(i) {
    return e(+i);
  }, r;
}(0);
function Ps(e, t, r) {
  var i = e._x1, a = e._y1, n = e._x2, o = e._y2;
  if (e._l01_a > Po) {
    var s = 2 * e._l01_2a + 3 * e._l01_a * e._l12_a + e._l12_2a, l = 3 * e._l01_a * (e._l01_a + e._l12_a);
    i = (i * s - e._x0 * e._l12_2a + e._x2 * e._l01_2a) / l, a = (a * s - e._y0 * e._l12_2a + e._y2 * e._l01_2a) / l;
  }
  if (e._l23_a > Po) {
    var c = 2 * e._l23_2a + 3 * e._l23_a * e._l12_a + e._l12_2a, h = 3 * e._l23_a * (e._l23_a + e._l12_a);
    n = (n * c + e._x1 * e._l23_2a - t * e._l12_2a) / h, o = (o * c + e._y1 * e._l23_2a - r * e._l12_2a) / h;
  }
  e._context.bezierCurveTo(i, a, n, o, e._x2, e._y2);
}
function ku(e, t) {
  this._context = e, this._alpha = t;
}
ku.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._x2 = this._y0 = this._y1 = this._y2 = NaN, this._l01_a = this._l12_a = this._l23_a = this._l01_2a = this._l12_2a = this._l23_2a = this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 2:
        this._context.lineTo(this._x2, this._y2);
        break;
      case 3:
        this.point(this._x2, this._y2);
        break;
    }
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    if (e = +e, t = +t, this._point) {
      var r = this._x2 - e, i = this._y2 - t;
      this._l23_a = Math.sqrt(this._l23_2a = Math.pow(r * r + i * i, this._alpha));
    }
    switch (this._point) {
      case 0:
        this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
        break;
      case 1:
        this._point = 2;
        break;
      case 2:
        this._point = 3;
      default:
        Ps(this, e, t);
        break;
    }
    this._l01_a = this._l12_a, this._l12_a = this._l23_a, this._l01_2a = this._l12_2a, this._l12_2a = this._l23_2a, this._x0 = this._x1, this._x1 = this._x2, this._x2 = e, this._y0 = this._y1, this._y1 = this._y2, this._y2 = t;
  }
};
const wu = function e(t) {
  function r(i) {
    return t ? new ku(i, t) : new Os(i, 0);
  }
  return r.alpha = function(i) {
    return e(+i);
  }, r;
}(0.5);
function _u(e, t) {
  this._context = e, this._alpha = t;
}
_u.prototype = {
  areaStart: ve,
  areaEnd: ve,
  lineStart: function() {
    this._x0 = this._x1 = this._x2 = this._x3 = this._x4 = this._x5 = this._y0 = this._y1 = this._y2 = this._y3 = this._y4 = this._y5 = NaN, this._l01_a = this._l12_a = this._l23_a = this._l01_2a = this._l12_2a = this._l23_2a = this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 1: {
        this._context.moveTo(this._x3, this._y3), this._context.closePath();
        break;
      }
      case 2: {
        this._context.lineTo(this._x3, this._y3), this._context.closePath();
        break;
      }
      case 3: {
        this.point(this._x3, this._y3), this.point(this._x4, this._y4), this.point(this._x5, this._y5);
        break;
      }
    }
  },
  point: function(e, t) {
    if (e = +e, t = +t, this._point) {
      var r = this._x2 - e, i = this._y2 - t;
      this._l23_a = Math.sqrt(this._l23_2a = Math.pow(r * r + i * i, this._alpha));
    }
    switch (this._point) {
      case 0:
        this._point = 1, this._x3 = e, this._y3 = t;
        break;
      case 1:
        this._point = 2, this._context.moveTo(this._x4 = e, this._y4 = t);
        break;
      case 2:
        this._point = 3, this._x5 = e, this._y5 = t;
        break;
      default:
        Ps(this, e, t);
        break;
    }
    this._l01_a = this._l12_a, this._l12_a = this._l23_a, this._l01_2a = this._l12_2a, this._l12_2a = this._l23_2a, this._x0 = this._x1, this._x1 = this._x2, this._x2 = e, this._y0 = this._y1, this._y1 = this._y2, this._y2 = t;
  }
};
const Hb = function e(t) {
  function r(i) {
    return t ? new _u(i, t) : new Ds(i, 0);
  }
  return r.alpha = function(i) {
    return e(+i);
  }, r;
}(0.5);
function vu(e, t) {
  this._context = e, this._alpha = t;
}
vu.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._x2 = this._y0 = this._y1 = this._y2 = NaN, this._l01_a = this._l12_a = this._l23_a = this._l01_2a = this._l12_2a = this._l23_2a = this._point = 0;
  },
  lineEnd: function() {
    (this._line || this._line !== 0 && this._point === 3) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    if (e = +e, t = +t, this._point) {
      var r = this._x2 - e, i = this._y2 - t;
      this._l23_a = Math.sqrt(this._l23_2a = Math.pow(r * r + i * i, this._alpha));
    }
    switch (this._point) {
      case 0:
        this._point = 1;
        break;
      case 1:
        this._point = 2;
        break;
      case 2:
        this._point = 3, this._line ? this._context.lineTo(this._x2, this._y2) : this._context.moveTo(this._x2, this._y2);
        break;
      case 3:
        this._point = 4;
      default:
        Ps(this, e, t);
        break;
    }
    this._l01_a = this._l12_a, this._l12_a = this._l23_a, this._l01_2a = this._l12_2a, this._l12_2a = this._l23_2a, this._x0 = this._x1, this._x1 = this._x2, this._x2 = e, this._y0 = this._y1, this._y1 = this._y2, this._y2 = t;
  }
};
const jb = function e(t) {
  function r(i) {
    return t ? new vu(i, t) : new Rs(i, 0);
  }
  return r.alpha = function(i) {
    return e(+i);
  }, r;
}(0.5);
function Su(e) {
  this._context = e;
}
Su.prototype = {
  areaStart: ve,
  areaEnd: ve,
  lineStart: function() {
    this._point = 0;
  },
  lineEnd: function() {
    this._point && this._context.closePath();
  },
  point: function(e, t) {
    e = +e, t = +t, this._point ? this._context.lineTo(e, t) : (this._point = 1, this._context.moveTo(e, t));
  }
};
function Yb(e) {
  return new Su(e);
}
function No(e) {
  return e < 0 ? -1 : 1;
}
function zo(e, t, r) {
  var i = e._x1 - e._x0, a = t - e._x1, n = (e._y1 - e._y0) / (i || a < 0 && -0), o = (r - e._y1) / (a || i < 0 && -0), s = (n * a + o * i) / (i + a);
  return (No(n) + No(o)) * Math.min(Math.abs(n), Math.abs(o), 0.5 * Math.abs(s)) || 0;
}
function qo(e, t) {
  var r = e._x1 - e._x0;
  return r ? (3 * (e._y1 - e._y0) / r - t) / 2 : t;
}
function en(e, t, r) {
  var i = e._x0, a = e._y0, n = e._x1, o = e._y1, s = (n - i) / 3;
  e._context.bezierCurveTo(i + s, a + s * t, n - s, o - s * r, n, o);
}
function la(e) {
  this._context = e;
}
la.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._y0 = this._y1 = this._t0 = NaN, this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 2:
        this._context.lineTo(this._x1, this._y1);
        break;
      case 3:
        en(this, this._t0, qo(this, this._t0));
        break;
    }
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    var r = NaN;
    if (e = +e, t = +t, !(e === this._x1 && t === this._y1)) {
      switch (this._point) {
        case 0:
          this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
          break;
        case 1:
          this._point = 2;
          break;
        case 2:
          this._point = 3, en(this, qo(this, r = zo(this, e, t)), r);
          break;
        default:
          en(this, this._t0, r = zo(this, e, t));
          break;
      }
      this._x0 = this._x1, this._x1 = e, this._y0 = this._y1, this._y1 = t, this._t0 = r;
    }
  }
};
function Tu(e) {
  this._context = new Bu(e);
}
(Tu.prototype = Object.create(la.prototype)).point = function(e, t) {
  la.prototype.point.call(this, t, e);
};
function Bu(e) {
  this._context = e;
}
Bu.prototype = {
  moveTo: function(e, t) {
    this._context.moveTo(t, e);
  },
  closePath: function() {
    this._context.closePath();
  },
  lineTo: function(e, t) {
    this._context.lineTo(t, e);
  },
  bezierCurveTo: function(e, t, r, i, a, n) {
    this._context.bezierCurveTo(t, e, i, r, n, a);
  }
};
function Lu(e) {
  return new la(e);
}
function Mu(e) {
  return new Tu(e);
}
function $u(e) {
  this._context = e;
}
$u.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x = [], this._y = [];
  },
  lineEnd: function() {
    var e = this._x, t = this._y, r = e.length;
    if (r)
      if (this._line ? this._context.lineTo(e[0], t[0]) : this._context.moveTo(e[0], t[0]), r === 2)
        this._context.lineTo(e[1], t[1]);
      else
        for (var i = Wo(e), a = Wo(t), n = 0, o = 1; o < r; ++n, ++o)
          this._context.bezierCurveTo(i[0][n], a[0][n], i[1][n], a[1][n], e[o], t[o]);
    (this._line || this._line !== 0 && r === 1) && this._context.closePath(), this._line = 1 - this._line, this._x = this._y = null;
  },
  point: function(e, t) {
    this._x.push(+e), this._y.push(+t);
  }
};
function Wo(e) {
  var t, r = e.length - 1, i, a = new Array(r), n = new Array(r), o = new Array(r);
  for (a[0] = 0, n[0] = 2, o[0] = e[0] + 2 * e[1], t = 1; t < r - 1; ++t) a[t] = 1, n[t] = 4, o[t] = 4 * e[t] + 2 * e[t + 1];
  for (a[r - 1] = 2, n[r - 1] = 7, o[r - 1] = 8 * e[r - 1] + e[r], t = 1; t < r; ++t) i = a[t] / n[t - 1], n[t] -= i, o[t] -= i * o[t - 1];
  for (a[r - 1] = o[r - 1] / n[r - 1], t = r - 2; t >= 0; --t) a[t] = (o[t] - a[t + 1]) / n[t];
  for (n[r - 1] = (e[r] + a[r - 1]) / 2, t = 0; t < r - 1; ++t) n[t] = 2 * e[t + 1] - a[t + 1];
  return [a, n];
}
function Au(e) {
  return new $u(e);
}
function Da(e, t) {
  this._context = e, this._t = t;
}
Da.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x = this._y = NaN, this._point = 0;
  },
  lineEnd: function() {
    0 < this._t && this._t < 1 && this._point === 2 && this._context.lineTo(this._x, this._y), (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line >= 0 && (this._t = 1 - this._t, this._line = 1 - this._line);
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
        break;
      case 1:
        this._point = 2;
      default: {
        if (this._t <= 0)
          this._context.lineTo(this._x, t), this._context.lineTo(e, t);
        else {
          var r = this._x * (1 - this._t) + e * this._t;
          this._context.lineTo(r, this._y), this._context.lineTo(r, t);
        }
        break;
      }
    }
    this._x = e, this._y = t;
  }
};
function Fu(e) {
  return new Da(e, 0.5);
}
function Eu(e) {
  return new Da(e, 0);
}
function Ou(e) {
  return new Da(e, 1);
}
function Wr(e, t, r) {
  this.k = e, this.x = t, this.y = r;
}
Wr.prototype = {
  constructor: Wr,
  scale: function(e) {
    return e === 1 ? this : new Wr(this.k * e, this.x, this.y);
  },
  translate: function(e, t) {
    return e === 0 & t === 0 ? this : new Wr(this.k, this.x + this.k * e, this.y + this.k * t);
  },
  apply: function(e) {
    return [e[0] * this.k + this.x, e[1] * this.k + this.y];
  },
  applyX: function(e) {
    return e * this.k + this.x;
  },
  applyY: function(e) {
    return e * this.k + this.y;
  },
  invert: function(e) {
    return [(e[0] - this.x) / this.k, (e[1] - this.y) / this.k];
  },
  invertX: function(e) {
    return (e - this.x) / this.k;
  },
  invertY: function(e) {
    return (e - this.y) / this.k;
  },
  rescaleX: function(e) {
    return e.copy().domain(e.range().map(this.invertX, this).map(e.invert, e));
  },
  rescaleY: function(e) {
    return e.copy().domain(e.range().map(this.invertY, this).map(e.invert, e));
  },
  toString: function() {
    return "translate(" + this.x + "," + this.y + ") scale(" + this.k + ")";
  }
};
Wr.prototype;
var Du = typeof global == "object" && global && global.Object === Object && global, Gb = typeof self == "object" && self && self.Object === Object && self, ae = Du || Gb || Function("return this")(), ca = ae.Symbol, Ru = Object.prototype, Ub = Ru.hasOwnProperty, Xb = Ru.toString, Fr = ca ? ca.toStringTag : void 0;
function Vb(e) {
  var t = Ub.call(e, Fr), r = e[Fr];
  try {
    e[Fr] = void 0;
    var i = !0;
  } catch {
  }
  var a = Xb.call(e);
  return i && (t ? e[Fr] = r : delete e[Fr]), a;
}
var Zb = Object.prototype, Kb = Zb.toString;
function Qb(e) {
  return Kb.call(e);
}
var Jb = "[object Null]", t1 = "[object Undefined]", Ho = ca ? ca.toStringTag : void 0;
function Tr(e) {
  return e == null ? e === void 0 ? t1 : Jb : Ho && Ho in Object(e) ? Vb(e) : Qb(e);
}
function Ue(e) {
  var t = typeof e;
  return e != null && (t == "object" || t == "function");
}
var e1 = "[object AsyncFunction]", r1 = "[object Function]", i1 = "[object GeneratorFunction]", a1 = "[object Proxy]";
function Is(e) {
  if (!Ue(e))
    return !1;
  var t = Tr(e);
  return t == r1 || t == i1 || t == e1 || t == a1;
}
var rn = ae["__core-js_shared__"], jo = function() {
  var e = /[^.]+$/.exec(rn && rn.keys && rn.keys.IE_PROTO || "");
  return e ? "Symbol(src)_1." + e : "";
}();
function n1(e) {
  return !!jo && jo in e;
}
var s1 = Function.prototype, o1 = s1.toString;
function Xe(e) {
  if (e != null) {
    try {
      return o1.call(e);
    } catch {
    }
    try {
      return e + "";
    } catch {
    }
  }
  return "";
}
var l1 = /[\\^$.*+?()[\]{}|]/g, c1 = /^\[object .+?Constructor\]$/, h1 = Function.prototype, u1 = Object.prototype, f1 = h1.toString, d1 = u1.hasOwnProperty, p1 = RegExp(
  "^" + f1.call(d1).replace(l1, "\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g, "$1.*?") + "$"
);
function g1(e) {
  if (!Ue(e) || n1(e))
    return !1;
  var t = Is(e) ? p1 : c1;
  return t.test(Xe(e));
}
function m1(e, t) {
  return e == null ? void 0 : e[t];
}
function Ve(e, t) {
  var r = m1(e, t);
  return g1(r) ? r : void 0;
}
var ni = Ve(Object, "create");
function y1() {
  this.__data__ = ni ? ni(null) : {}, this.size = 0;
}
function x1(e) {
  var t = this.has(e) && delete this.__data__[e];
  return this.size -= t ? 1 : 0, t;
}
var b1 = "__lodash_hash_undefined__", C1 = Object.prototype, k1 = C1.hasOwnProperty;
function w1(e) {
  var t = this.__data__;
  if (ni) {
    var r = t[e];
    return r === b1 ? void 0 : r;
  }
  return k1.call(t, e) ? t[e] : void 0;
}
var _1 = Object.prototype, v1 = _1.hasOwnProperty;
function S1(e) {
  var t = this.__data__;
  return ni ? t[e] !== void 0 : v1.call(t, e);
}
var T1 = "__lodash_hash_undefined__";
function B1(e, t) {
  var r = this.__data__;
  return this.size += this.has(e) ? 0 : 1, r[e] = ni && t === void 0 ? T1 : t, this;
}
function je(e) {
  var t = -1, r = e == null ? 0 : e.length;
  for (this.clear(); ++t < r; ) {
    var i = e[t];
    this.set(i[0], i[1]);
  }
}
je.prototype.clear = y1;
je.prototype.delete = x1;
je.prototype.get = w1;
je.prototype.has = S1;
je.prototype.set = B1;
function L1() {
  this.__data__ = [], this.size = 0;
}
function Ra(e, t) {
  return e === t || e !== e && t !== t;
}
function Pa(e, t) {
  for (var r = e.length; r--; )
    if (Ra(e[r][0], t))
      return r;
  return -1;
}
var M1 = Array.prototype, $1 = M1.splice;
function A1(e) {
  var t = this.__data__, r = Pa(t, e);
  if (r < 0)
    return !1;
  var i = t.length - 1;
  return r == i ? t.pop() : $1.call(t, r, 1), --this.size, !0;
}
function F1(e) {
  var t = this.__data__, r = Pa(t, e);
  return r < 0 ? void 0 : t[r][1];
}
function E1(e) {
  return Pa(this.__data__, e) > -1;
}
function O1(e, t) {
  var r = this.__data__, i = Pa(r, e);
  return i < 0 ? (++this.size, r.push([e, t])) : r[i][1] = t, this;
}
function ge(e) {
  var t = -1, r = e == null ? 0 : e.length;
  for (this.clear(); ++t < r; ) {
    var i = e[t];
    this.set(i[0], i[1]);
  }
}
ge.prototype.clear = L1;
ge.prototype.delete = A1;
ge.prototype.get = F1;
ge.prototype.has = E1;
ge.prototype.set = O1;
var si = Ve(ae, "Map");
function D1() {
  this.size = 0, this.__data__ = {
    hash: new je(),
    map: new (si || ge)(),
    string: new je()
  };
}
function R1(e) {
  var t = typeof e;
  return t == "string" || t == "number" || t == "symbol" || t == "boolean" ? e !== "__proto__" : e === null;
}
function Ia(e, t) {
  var r = e.__data__;
  return R1(t) ? r[typeof t == "string" ? "string" : "hash"] : r.map;
}
function P1(e) {
  var t = Ia(this, e).delete(e);
  return this.size -= t ? 1 : 0, t;
}
function I1(e) {
  return Ia(this, e).get(e);
}
function N1(e) {
  return Ia(this, e).has(e);
}
function z1(e, t) {
  var r = Ia(this, e), i = r.size;
  return r.set(e, t), this.size += r.size == i ? 0 : 1, this;
}
function Be(e) {
  var t = -1, r = e == null ? 0 : e.length;
  for (this.clear(); ++t < r; ) {
    var i = e[t];
    this.set(i[0], i[1]);
  }
}
Be.prototype.clear = D1;
Be.prototype.delete = P1;
Be.prototype.get = I1;
Be.prototype.has = N1;
Be.prototype.set = z1;
var q1 = "Expected a function";
function pi(e, t) {
  if (typeof e != "function" || t != null && typeof t != "function")
    throw new TypeError(q1);
  var r = function() {
    var i = arguments, a = t ? t.apply(this, i) : i[0], n = r.cache;
    if (n.has(a))
      return n.get(a);
    var o = e.apply(this, i);
    return r.cache = n.set(a, o) || n, o;
  };
  return r.cache = new (pi.Cache || Be)(), r;
}
pi.Cache = Be;
function W1() {
  this.__data__ = new ge(), this.size = 0;
}
function H1(e) {
  var t = this.__data__, r = t.delete(e);
  return this.size = t.size, r;
}
function j1(e) {
  return this.__data__.get(e);
}
function Y1(e) {
  return this.__data__.has(e);
}
var G1 = 200;
function U1(e, t) {
  var r = this.__data__;
  if (r instanceof ge) {
    var i = r.__data__;
    if (!si || i.length < G1 - 1)
      return i.push([e, t]), this.size = ++r.size, this;
    r = this.__data__ = new Be(i);
  }
  return r.set(e, t), this.size = r.size, this;
}
function Br(e) {
  var t = this.__data__ = new ge(e);
  this.size = t.size;
}
Br.prototype.clear = W1;
Br.prototype.delete = H1;
Br.prototype.get = j1;
Br.prototype.has = Y1;
Br.prototype.set = U1;
var ha = function() {
  try {
    var e = Ve(Object, "defineProperty");
    return e({}, "", {}), e;
  } catch {
  }
}();
function Ns(e, t, r) {
  t == "__proto__" && ha ? ha(e, t, {
    configurable: !0,
    enumerable: !0,
    value: r,
    writable: !0
  }) : e[t] = r;
}
function jn(e, t, r) {
  (r !== void 0 && !Ra(e[t], r) || r === void 0 && !(t in e)) && Ns(e, t, r);
}
function X1(e) {
  return function(t, r, i) {
    for (var a = -1, n = Object(t), o = i(t), s = o.length; s--; ) {
      var l = o[++a];
      if (r(n[l], l, n) === !1)
        break;
    }
    return t;
  };
}
var V1 = X1(), Pu = typeof exports == "object" && exports && !exports.nodeType && exports, Yo = Pu && typeof module == "object" && module && !module.nodeType && module, Z1 = Yo && Yo.exports === Pu, Go = Z1 ? ae.Buffer : void 0, Uo = Go ? Go.allocUnsafe : void 0;
function K1(e, t) {
  if (t)
    return e.slice();
  var r = e.length, i = Uo ? Uo(r) : new e.constructor(r);
  return e.copy(i), i;
}
var Xo = ae.Uint8Array;
function Q1(e) {
  var t = new e.constructor(e.byteLength);
  return new Xo(t).set(new Xo(e)), t;
}
function J1(e, t) {
  var r = t ? Q1(e.buffer) : e.buffer;
  return new e.constructor(r, e.byteOffset, e.length);
}
function t2(e, t) {
  var r = -1, i = e.length;
  for (t || (t = Array(i)); ++r < i; )
    t[r] = e[r];
  return t;
}
var Vo = Object.create, e2 = /* @__PURE__ */ function() {
  function e() {
  }
  return function(t) {
    if (!Ue(t))
      return {};
    if (Vo)
      return Vo(t);
    e.prototype = t;
    var r = new e();
    return e.prototype = void 0, r;
  };
}();
function Iu(e, t) {
  return function(r) {
    return e(t(r));
  };
}
var Nu = Iu(Object.getPrototypeOf, Object), r2 = Object.prototype;
function Na(e) {
  var t = e && e.constructor, r = typeof t == "function" && t.prototype || r2;
  return e === r;
}
function i2(e) {
  return typeof e.constructor == "function" && !Na(e) ? e2(Nu(e)) : {};
}
function gi(e) {
  return e != null && typeof e == "object";
}
var a2 = "[object Arguments]";
function Zo(e) {
  return gi(e) && Tr(e) == a2;
}
var zu = Object.prototype, n2 = zu.hasOwnProperty, s2 = zu.propertyIsEnumerable, ua = Zo(/* @__PURE__ */ function() {
  return arguments;
}()) ? Zo : function(e) {
  return gi(e) && n2.call(e, "callee") && !s2.call(e, "callee");
}, fa = Array.isArray, o2 = 9007199254740991;
function qu(e) {
  return typeof e == "number" && e > -1 && e % 1 == 0 && e <= o2;
}
function za(e) {
  return e != null && qu(e.length) && !Is(e);
}
function l2(e) {
  return gi(e) && za(e);
}
function c2() {
  return !1;
}
var Wu = typeof exports == "object" && exports && !exports.nodeType && exports, Ko = Wu && typeof module == "object" && module && !module.nodeType && module, h2 = Ko && Ko.exports === Wu, Qo = h2 ? ae.Buffer : void 0, u2 = Qo ? Qo.isBuffer : void 0, zs = u2 || c2, f2 = "[object Object]", d2 = Function.prototype, p2 = Object.prototype, Hu = d2.toString, g2 = p2.hasOwnProperty, m2 = Hu.call(Object);
function y2(e) {
  if (!gi(e) || Tr(e) != f2)
    return !1;
  var t = Nu(e);
  if (t === null)
    return !0;
  var r = g2.call(t, "constructor") && t.constructor;
  return typeof r == "function" && r instanceof r && Hu.call(r) == m2;
}
var x2 = "[object Arguments]", b2 = "[object Array]", C2 = "[object Boolean]", k2 = "[object Date]", w2 = "[object Error]", _2 = "[object Function]", v2 = "[object Map]", S2 = "[object Number]", T2 = "[object Object]", B2 = "[object RegExp]", L2 = "[object Set]", M2 = "[object String]", $2 = "[object WeakMap]", A2 = "[object ArrayBuffer]", F2 = "[object DataView]", E2 = "[object Float32Array]", O2 = "[object Float64Array]", D2 = "[object Int8Array]", R2 = "[object Int16Array]", P2 = "[object Int32Array]", I2 = "[object Uint8Array]", N2 = "[object Uint8ClampedArray]", z2 = "[object Uint16Array]", q2 = "[object Uint32Array]", ct = {};
ct[E2] = ct[O2] = ct[D2] = ct[R2] = ct[P2] = ct[I2] = ct[N2] = ct[z2] = ct[q2] = !0;
ct[x2] = ct[b2] = ct[A2] = ct[C2] = ct[F2] = ct[k2] = ct[w2] = ct[_2] = ct[v2] = ct[S2] = ct[T2] = ct[B2] = ct[L2] = ct[M2] = ct[$2] = !1;
function W2(e) {
  return gi(e) && qu(e.length) && !!ct[Tr(e)];
}
function H2(e) {
  return function(t) {
    return e(t);
  };
}
var ju = typeof exports == "object" && exports && !exports.nodeType && exports, Vr = ju && typeof module == "object" && module && !module.nodeType && module, j2 = Vr && Vr.exports === ju, an = j2 && Du.process, Jo = function() {
  try {
    var e = Vr && Vr.require && Vr.require("util").types;
    return e || an && an.binding && an.binding("util");
  } catch {
  }
}(), tl = Jo && Jo.isTypedArray, qs = tl ? H2(tl) : W2;
function Yn(e, t) {
  if (!(t === "constructor" && typeof e[t] == "function") && t != "__proto__")
    return e[t];
}
var Y2 = Object.prototype, G2 = Y2.hasOwnProperty;
function U2(e, t, r) {
  var i = e[t];
  (!(G2.call(e, t) && Ra(i, r)) || r === void 0 && !(t in e)) && Ns(e, t, r);
}
function X2(e, t, r, i) {
  var a = !r;
  r || (r = {});
  for (var n = -1, o = t.length; ++n < o; ) {
    var s = t[n], l = void 0;
    l === void 0 && (l = e[s]), a ? Ns(r, s, l) : U2(r, s, l);
  }
  return r;
}
function V2(e, t) {
  for (var r = -1, i = Array(e); ++r < e; )
    i[r] = t(r);
  return i;
}
var Z2 = 9007199254740991, K2 = /^(?:0|[1-9]\d*)$/;
function Yu(e, t) {
  var r = typeof e;
  return t = t ?? Z2, !!t && (r == "number" || r != "symbol" && K2.test(e)) && e > -1 && e % 1 == 0 && e < t;
}
var Q2 = Object.prototype, J2 = Q2.hasOwnProperty;
function tC(e, t) {
  var r = fa(e), i = !r && ua(e), a = !r && !i && zs(e), n = !r && !i && !a && qs(e), o = r || i || a || n, s = o ? V2(e.length, String) : [], l = s.length;
  for (var c in e)
    (t || J2.call(e, c)) && !(o && // Safari 9 has enumerable `arguments.length` in strict mode.
    (c == "length" || // Node.js 0.10 has enumerable non-index properties on buffers.
    a && (c == "offset" || c == "parent") || // PhantomJS 2 has enumerable non-index properties on typed arrays.
    n && (c == "buffer" || c == "byteLength" || c == "byteOffset") || // Skip index properties.
    Yu(c, l))) && s.push(c);
  return s;
}
function eC(e) {
  var t = [];
  if (e != null)
    for (var r in Object(e))
      t.push(r);
  return t;
}
var rC = Object.prototype, iC = rC.hasOwnProperty;
function aC(e) {
  if (!Ue(e))
    return eC(e);
  var t = Na(e), r = [];
  for (var i in e)
    i == "constructor" && (t || !iC.call(e, i)) || r.push(i);
  return r;
}
function Gu(e) {
  return za(e) ? tC(e, !0) : aC(e);
}
function nC(e) {
  return X2(e, Gu(e));
}
function sC(e, t, r, i, a, n, o) {
  var s = Yn(e, r), l = Yn(t, r), c = o.get(l);
  if (c) {
    jn(e, r, c);
    return;
  }
  var h = n ? n(s, l, r + "", e, t, o) : void 0, u = h === void 0;
  if (u) {
    var f = fa(l), d = !f && zs(l), g = !f && !d && qs(l);
    h = l, f || d || g ? fa(s) ? h = s : l2(s) ? h = t2(s) : d ? (u = !1, h = K1(l, !0)) : g ? (u = !1, h = J1(l, !0)) : h = [] : y2(l) || ua(l) ? (h = s, ua(s) ? h = nC(s) : (!Ue(s) || Is(s)) && (h = i2(l))) : u = !1;
  }
  u && (o.set(l, h), a(h, l, i, n, o), o.delete(l)), jn(e, r, h);
}
function Uu(e, t, r, i, a) {
  e !== t && V1(t, function(n, o) {
    if (a || (a = new Br()), Ue(n))
      sC(e, t, o, r, Uu, i, a);
    else {
      var s = i ? i(Yn(e, o), n, o + "", e, t, a) : void 0;
      s === void 0 && (s = n), jn(e, o, s);
    }
  }, Gu);
}
function Xu(e) {
  return e;
}
function oC(e, t, r) {
  switch (r.length) {
    case 0:
      return e.call(t);
    case 1:
      return e.call(t, r[0]);
    case 2:
      return e.call(t, r[0], r[1]);
    case 3:
      return e.call(t, r[0], r[1], r[2]);
  }
  return e.apply(t, r);
}
var el = Math.max;
function lC(e, t, r) {
  return t = el(t === void 0 ? e.length - 1 : t, 0), function() {
    for (var i = arguments, a = -1, n = el(i.length - t, 0), o = Array(n); ++a < n; )
      o[a] = i[t + a];
    a = -1;
    for (var s = Array(t + 1); ++a < t; )
      s[a] = i[a];
    return s[t] = r(o), oC(e, this, s);
  };
}
function cC(e) {
  return function() {
    return e;
  };
}
var hC = ha ? function(e, t) {
  return ha(e, "toString", {
    configurable: !0,
    enumerable: !1,
    value: cC(t),
    writable: !0
  });
} : Xu, uC = 800, fC = 16, dC = Date.now;
function pC(e) {
  var t = 0, r = 0;
  return function() {
    var i = dC(), a = fC - (i - r);
    if (r = i, a > 0) {
      if (++t >= uC)
        return arguments[0];
    } else
      t = 0;
    return e.apply(void 0, arguments);
  };
}
var gC = pC(hC);
function mC(e, t) {
  return gC(lC(e, t, Xu), e + "");
}
function yC(e, t, r) {
  if (!Ue(r))
    return !1;
  var i = typeof t;
  return (i == "number" ? za(r) && Yu(t, r.length) : i == "string" && t in r) ? Ra(r[t], e) : !1;
}
function xC(e) {
  return mC(function(t, r) {
    var i = -1, a = r.length, n = a > 1 ? r[a - 1] : void 0, o = a > 2 ? r[2] : void 0;
    for (n = e.length > 3 && typeof n == "function" ? (a--, n) : void 0, o && yC(r[0], r[1], o) && (n = a < 3 ? void 0 : n, a = 1), t = Object(t); ++i < a; ) {
      var s = r[i];
      s && e(t, s, i, n);
    }
    return t;
  });
}
var bC = xC(function(e, t, r) {
  Uu(e, t, r);
}), CC = "​", kC = {
  curveBasis: Ri,
  curveBasisClosed: Ib,
  curveBasisOpen: Nb,
  curveBumpX: gu,
  curveBumpY: mu,
  curveBundle: zb,
  curveCardinalClosed: qb,
  curveCardinalOpen: Wb,
  curveCardinal: Cu,
  curveCatmullRomClosed: Hb,
  curveCatmullRomOpen: jb,
  curveCatmullRom: wu,
  curveLinear: na,
  curveLinearClosed: Yb,
  curveMonotoneX: Lu,
  curveMonotoneY: Mu,
  curveNatural: Au,
  curveStep: Fu,
  curveStepAfter: Ou,
  curveStepBefore: Eu
}, wC = /\s*(?:(\w+)(?=:):|(\w+))\s*(?:(\w+)|((?:(?!}%{2}).|\r?\n)*))?\s*(?:}%{2})?/gi, _C = /* @__PURE__ */ p(function(e, t) {
  const r = Vu(e, /(?:init\b)|(?:initialize\b)/);
  let i = {};
  if (Array.isArray(r)) {
    const o = r.map((s) => s.args);
    Hi(o), i = vt(i, [...o]);
  } else
    i = r.args;
  if (!i)
    return;
  let a = us(e, t);
  const n = "config";
  return i[n] !== void 0 && (a === "flowchart-v2" && (a = "flowchart"), i[a] = i[n], delete i[n]), i;
}, "detectInit"), Vu = /* @__PURE__ */ p(function(e, t = null) {
  var r, i;
  try {
    const a = new RegExp(
      `[%]{2}(?![{]${wC.source})(?=[}][%]{2}).*
`,
      "ig"
    );
    e = e.trim().replace(a, "").replace(/'/gm, '"'), F.debug(
      `Detecting diagram directive${t !== null ? " type:" + t : ""} based on the text:${e}`
    );
    let n;
    const o = [];
    for (; (n = Ur.exec(e)) !== null; )
      if (n.index === Ur.lastIndex && Ur.lastIndex++, n && !t || t && ((r = n[1]) != null && r.match(t)) || t && ((i = n[2]) != null && i.match(t))) {
        const s = n[1] ? n[1] : n[2], l = n[3] ? n[3].trim() : n[4] ? JSON.parse(n[4].trim()) : null;
        o.push({ type: s, args: l });
      }
    return o.length === 0 ? { type: e, args: null } : o.length === 1 ? o[0] : o;
  } catch (a) {
    return F.error(
      `ERROR: ${a.message} - Unable to parse directive type: '${t}' based on the text: '${e}'`
    ), { type: void 0, args: null };
  }
}, "detectDirective"), vC = /* @__PURE__ */ p(function(e) {
  return e.replace(Ur, "");
}, "removeDirectives"), SC = /* @__PURE__ */ p(function(e, t) {
  for (const [r, i] of t.entries())
    if (i.match(e))
      return r;
  return -1;
}, "isSubstringInArray");
function Ws(e, t) {
  if (!e)
    return t;
  const r = `curve${e.charAt(0).toUpperCase() + e.slice(1)}`;
  return kC[r] ?? t;
}
p(Ws, "interpolateToCurve");
function Zu(e, t) {
  const r = e.trim();
  if (r)
    return t.securityLevel !== "loose" ? Nh(r) : r;
}
p(Zu, "formatUrl");
var TC = /* @__PURE__ */ p((e, ...t) => {
  const r = e.split("."), i = r.length - 1, a = r[i];
  let n = window;
  for (let o = 0; o < i; o++)
    if (n = n[r[o]], !n) {
      F.error(`Function name: ${e} not found in window`);
      return;
    }
  n[a](...t);
}, "runFunc");
function Hs(e, t) {
  return !e || !t ? 0 : Math.sqrt(Math.pow(t.x - e.x, 2) + Math.pow(t.y - e.y, 2));
}
p(Hs, "distance");
function Ku(e) {
  let t, r = 0;
  e.forEach((a) => {
    r += Hs(a, t), t = a;
  });
  const i = r / 2;
  return js(e, i);
}
p(Ku, "traverseEdge");
function Qu(e) {
  return e.length === 1 ? e[0] : Ku(e);
}
p(Qu, "calcLabelPosition");
var rl = /* @__PURE__ */ p((e, t = 2) => {
  const r = Math.pow(10, t);
  return Math.round(e * r) / r;
}, "roundNumber"), js = /* @__PURE__ */ p((e, t) => {
  let r, i = t;
  for (const a of e) {
    if (r) {
      const n = Hs(a, r);
      if (n === 0)
        return r;
      if (n < i)
        i -= n;
      else {
        const o = i / n;
        if (o <= 0)
          return r;
        if (o >= 1)
          return { x: a.x, y: a.y };
        if (o > 0 && o < 1)
          return {
            x: rl((1 - o) * r.x + o * a.x, 5),
            y: rl((1 - o) * r.y + o * a.y, 5)
          };
      }
    }
    r = a;
  }
  throw new Error("Could not find a suitable point for the given distance");
}, "calculatePoint"), BC = /* @__PURE__ */ p((e, t, r) => {
  F.info(`our points ${JSON.stringify(t)}`), t[0] !== r && (t = t.reverse());
  const a = js(t, 25), n = e ? 10 : 5, o = Math.atan2(t[0].y - a.y, t[0].x - a.x), s = { x: 0, y: 0 };
  return s.x = Math.sin(o) * n + (t[0].x + a.x) / 2, s.y = -Math.cos(o) * n + (t[0].y + a.y) / 2, s;
}, "calcCardinalityPosition");
function Ju(e, t, r) {
  const i = structuredClone(r);
  F.info("our points", i), t !== "start_left" && t !== "start_right" && i.reverse();
  const a = 25 + e, n = js(i, a), o = 10 + e * 0.5, s = Math.atan2(i[0].y - n.y, i[0].x - n.x), l = { x: 0, y: 0 };
  return t === "start_left" ? (l.x = Math.sin(s + Math.PI) * o + (i[0].x + n.x) / 2, l.y = -Math.cos(s + Math.PI) * o + (i[0].y + n.y) / 2) : t === "end_right" ? (l.x = Math.sin(s - Math.PI) * o + (i[0].x + n.x) / 2 - 5, l.y = -Math.cos(s - Math.PI) * o + (i[0].y + n.y) / 2 - 5) : t === "end_left" ? (l.x = Math.sin(s) * o + (i[0].x + n.x) / 2 - 5, l.y = -Math.cos(s) * o + (i[0].y + n.y) / 2 - 5) : (l.x = Math.sin(s) * o + (i[0].x + n.x) / 2, l.y = -Math.cos(s) * o + (i[0].y + n.y) / 2), l;
}
p(Ju, "calcTerminalLabelPosition");
function tf(e) {
  let t = "", r = "";
  for (const i of e)
    i !== void 0 && (i.startsWith("color:") || i.startsWith("text-align:") ? r = r + i + ";" : t = t + i + ";");
  return { style: t, labelStyle: r };
}
p(tf, "getStylesFromArray");
var il = 0, LC = /* @__PURE__ */ p(() => (il++, "id-" + Math.random().toString(36).substr(2, 12) + "-" + il), "generateId");
function ef(e) {
  let t = "";
  const r = "0123456789abcdef", i = r.length;
  for (let a = 0; a < e; a++)
    t += r.charAt(Math.floor(Math.random() * i));
  return t;
}
p(ef, "makeRandomHex");
var MC = /* @__PURE__ */ p((e) => ef(e.length), "random"), $C = /* @__PURE__ */ p(function() {
  return {
    x: 0,
    y: 0,
    fill: void 0,
    anchor: "start",
    style: "#666",
    width: 100,
    height: 100,
    textMargin: 0,
    rx: 0,
    ry: 0,
    valign: void 0,
    text: ""
  };
}, "getTextObj"), AC = /* @__PURE__ */ p(function(e, t) {
  const r = t.text.replace(vr.lineBreakRegex, " "), [, i] = qa(t.fontSize), a = e.append("text");
  a.attr("x", t.x), a.attr("y", t.y), a.style("text-anchor", t.anchor), a.style("font-family", t.fontFamily), a.style("font-size", i), a.style("font-weight", t.fontWeight), a.attr("fill", t.fill), t.class !== void 0 && a.attr("class", t.class);
  const n = a.append("tspan");
  return n.attr("x", t.x + t.textMargin * 2), n.attr("fill", t.fill), n.text(r), a;
}, "drawSimpleText"), FC = pi(
  (e, t, r) => {
    if (!e || (r = Object.assign(
      { fontSize: 12, fontWeight: 400, fontFamily: "Arial", joinWith: "<br/>" },
      r
    ), vr.lineBreakRegex.test(e)))
      return e;
    const i = e.split(" ").filter(Boolean), a = [];
    let n = "";
    return i.forEach((o, s) => {
      const l = pe(`${o} `, r), c = pe(n, r);
      if (l > t) {
        const { hyphenatedStrings: f, remainingWord: d } = EC(o, t, "-", r);
        a.push(n, ...f), n = d;
      } else c + l >= t ? (a.push(n), n = o) : n = [n, o].filter(Boolean).join(" ");
      s + 1 === i.length && a.push(n);
    }), a.filter((o) => o !== "").join(r.joinWith);
  },
  (e, t, r) => `${e}${t}${r.fontSize}${r.fontWeight}${r.fontFamily}${r.joinWith}`
), EC = pi(
  (e, t, r = "-", i) => {
    i = Object.assign(
      { fontSize: 12, fontWeight: 400, fontFamily: "Arial", margin: 0 },
      i
    );
    const a = [...e], n = [];
    let o = "";
    return a.forEach((s, l) => {
      const c = `${o}${s}`;
      if (pe(c, i) >= t) {
        const u = l + 1, f = a.length === u, d = `${c}${r}`;
        n.push(f ? c : d), o = "";
      } else
        o = c;
    }), { hyphenatedStrings: n, remainingWord: o };
  },
  (e, t, r = "-", i) => `${e}${t}${r}${i.fontSize}${i.fontWeight}${i.fontFamily}`
);
function rf(e, t) {
  return Ys(e, t).height;
}
p(rf, "calculateTextHeight");
function pe(e, t) {
  return Ys(e, t).width;
}
p(pe, "calculateTextWidth");
var Ys = pi(
  (e, t) => {
    const { fontSize: r = 12, fontFamily: i = "Arial", fontWeight: a = 400 } = t;
    if (!e)
      return { width: 0, height: 0 };
    const [, n] = qa(r), o = ["sans-serif", i], s = e.split(vr.lineBreakRegex), l = [], c = et("body");
    if (!c.remove)
      return { width: 0, height: 0, lineHeight: 0 };
    const h = c.append("svg");
    for (const f of o) {
      let d = 0;
      const g = { width: 0, height: 0, lineHeight: 0 };
      for (const m of s) {
        const y = $C();
        y.text = m || CC;
        const x = AC(h, y).style("font-size", n).style("font-weight", a).style("font-family", f), b = (x._groups || x)[0][0].getBBox();
        if (b.width === 0 && b.height === 0)
          throw new Error("svg element not in render tree");
        g.width = Math.round(Math.max(g.width, b.width)), d = Math.round(b.height), g.height += d, g.lineHeight = Math.round(Math.max(g.lineHeight, d));
      }
      l.push(g);
    }
    h.remove();
    const u = isNaN(l[1].height) || isNaN(l[1].width) || isNaN(l[1].lineHeight) || l[0].height > l[1].height && l[0].width > l[1].width && l[0].lineHeight > l[1].lineHeight ? 0 : 1;
    return l[u];
  },
  (e, t) => `${e}${t.fontSize}${t.fontWeight}${t.fontFamily}`
), dr, OC = (dr = class {
  constructor(t = !1, r) {
    this.count = 0, this.count = r ? r.length : 0, this.next = t ? () => this.count++ : () => Date.now();
  }
}, p(dr, "InitIDGenerator"), dr), _i, DC = /* @__PURE__ */ p(function(e) {
  return _i = _i || document.createElement("div"), e = escape(e).replace(/%26/g, "&").replace(/%23/g, "#").replace(/%3B/g, ";"), _i.innerHTML = e, unescape(_i.textContent);
}, "entityDecode");
function Gs(e) {
  return "str" in e;
}
p(Gs, "isDetailedError");
var RC = /* @__PURE__ */ p((e, t, r, i) => {
  var n;
  if (!i)
    return;
  const a = (n = e.node()) == null ? void 0 : n.getBBox();
  a && e.append("text").text(i).attr("text-anchor", "middle").attr("x", a.x + a.width / 2).attr("y", -r).attr("class", t);
}, "insertTitle"), qa = /* @__PURE__ */ p((e) => {
  if (typeof e == "number")
    return [e, e + "px"];
  const t = parseInt(e ?? "", 10);
  return Number.isNaN(t) ? [void 0, void 0] : e === String(t) ? [t, e + "px"] : [t, e];
}, "parseFontSize");
function Us(e, t) {
  return bC({}, e, t);
}
p(Us, "cleanAndMerge");
var Jt = {
  assignWithDepth: vt,
  wrapLabel: FC,
  calculateTextHeight: rf,
  calculateTextWidth: pe,
  calculateTextDimensions: Ys,
  cleanAndMerge: Us,
  detectInit: _C,
  detectDirective: Vu,
  isSubstringInArray: SC,
  interpolateToCurve: Ws,
  calcLabelPosition: Qu,
  calcCardinalityPosition: BC,
  calcTerminalLabelPosition: Ju,
  formatUrl: Zu,
  getStylesFromArray: tf,
  generateId: LC,
  random: MC,
  runFunc: TC,
  entityDecode: DC,
  insertTitle: RC,
  parseFontSize: qa,
  InitIDGenerator: OC
}, PC = /* @__PURE__ */ p(function(e) {
  let t = e;
  return t = t.replace(/style.*:\S*#.*;/g, function(r) {
    return r.substring(0, r.length - 1);
  }), t = t.replace(/classDef.*:\S*#.*;/g, function(r) {
    return r.substring(0, r.length - 1);
  }), t = t.replace(/#\w+;/g, function(r) {
    const i = r.substring(1, r.length - 1);
    return /^\+?\d+$/.test(i) ? "ﬂ°°" + i + "¶ß" : "ﬂ°" + i + "¶ß";
  }), t;
}, "encodeEntities"), Ze = /* @__PURE__ */ p(function(e) {
  return e.replace(/ﬂ°°/g, "&#").replace(/ﬂ°/g, "&").replace(/¶ß/g, ";");
}, "decodeEntities"), PT = /* @__PURE__ */ p((e, t, {
  counter: r = 0,
  prefix: i,
  suffix: a
}, n) => n || `${i ? `${i}_` : ""}${e}_${t}_${r}${a ? `_${a}` : ""}`, "getEdgeId");
function Et(e) {
  return e ?? null;
}
p(Et, "handleUndefinedAttr");
const IC = Object.freeze(
  {
    left: 0,
    top: 0,
    width: 16,
    height: 16
  }
), da = Object.freeze({
  rotate: 0,
  vFlip: !1,
  hFlip: !1
}), af = Object.freeze({
  ...IC,
  ...da
}), NC = Object.freeze({
  ...af,
  body: "",
  hidden: !1
}), zC = Object.freeze({
  width: null,
  height: null
}), qC = Object.freeze({
  // Dimensions
  ...zC,
  // Transformations
  ...da
}), WC = (e, t, r, i = "") => {
  const a = e.split(":");
  if (e.slice(0, 1) === "@") {
    if (a.length < 2 || a.length > 3)
      return null;
    i = a.shift().slice(1);
  }
  if (a.length > 3 || !a.length)
    return null;
  if (a.length > 1) {
    const s = a.pop(), l = a.pop(), c = {
      // Allow provider without '@': "provider:prefix:name"
      provider: a.length > 0 ? a[0] : i,
      prefix: l,
      name: s
    };
    return nn(c) ? c : null;
  }
  const n = a[0], o = n.split("-");
  if (o.length > 1) {
    const s = {
      provider: i,
      prefix: o.shift(),
      name: o.join("-")
    };
    return nn(s) ? s : null;
  }
  if (r && i === "") {
    const s = {
      provider: i,
      prefix: "",
      name: n
    };
    return nn(s, r) ? s : null;
  }
  return null;
}, nn = (e, t) => e ? !!// Check prefix: cannot be empty, unless allowSimpleName is enabled
// Check name: cannot be empty
((t && e.prefix === "" || e.prefix) && e.name) : !1;
function HC(e, t) {
  const r = {};
  !e.hFlip != !t.hFlip && (r.hFlip = !0), !e.vFlip != !t.vFlip && (r.vFlip = !0);
  const i = ((e.rotate || 0) + (t.rotate || 0)) % 4;
  return i && (r.rotate = i), r;
}
function al(e, t) {
  const r = HC(e, t);
  for (const i in NC)
    i in da ? i in e && !(i in r) && (r[i] = da[i]) : i in t ? r[i] = t[i] : i in e && (r[i] = e[i]);
  return r;
}
function jC(e, t) {
  const r = e.icons, i = e.aliases || /* @__PURE__ */ Object.create(null), a = /* @__PURE__ */ Object.create(null);
  function n(o) {
    if (r[o])
      return a[o] = [];
    if (!(o in a)) {
      a[o] = null;
      const s = i[o] && i[o].parent, l = s && n(s);
      l && (a[o] = [s].concat(l));
    }
    return a[o];
  }
  return (t || Object.keys(r).concat(Object.keys(i))).forEach(n), a;
}
function nl(e, t, r) {
  const i = e.icons, a = e.aliases || /* @__PURE__ */ Object.create(null);
  let n = {};
  function o(s) {
    n = al(
      i[s] || a[s],
      n
    );
  }
  return o(t), r.forEach(o), al(e, n);
}
function YC(e, t) {
  if (e.icons[t])
    return nl(e, t, []);
  const r = jC(e, [t])[t];
  return r ? nl(e, t, r) : null;
}
const GC = /(-?[0-9.]*[0-9]+[0-9.]*)/g, UC = /^-?[0-9.]*[0-9]+[0-9.]*$/g;
function sl(e, t, r) {
  if (t === 1)
    return e;
  if (r = r || 100, typeof e == "number")
    return Math.ceil(e * t * r) / r;
  if (typeof e != "string")
    return e;
  const i = e.split(GC);
  if (i === null || !i.length)
    return e;
  const a = [];
  let n = i.shift(), o = UC.test(n);
  for (; ; ) {
    if (o) {
      const s = parseFloat(n);
      isNaN(s) ? a.push(n) : a.push(Math.ceil(s * t * r) / r);
    } else
      a.push(n);
    if (n = i.shift(), n === void 0)
      return a.join("");
    o = !o;
  }
}
function XC(e, t = "defs") {
  let r = "";
  const i = e.indexOf("<" + t);
  for (; i >= 0; ) {
    const a = e.indexOf(">", i), n = e.indexOf("</" + t);
    if (a === -1 || n === -1)
      break;
    const o = e.indexOf(">", n);
    if (o === -1)
      break;
    r += e.slice(a + 1, n).trim(), e = e.slice(0, i).trim() + e.slice(o + 1);
  }
  return {
    defs: r,
    content: e
  };
}
function VC(e, t) {
  return e ? "<defs>" + e + "</defs>" + t : t;
}
function ZC(e, t, r) {
  const i = XC(e);
  return VC(i.defs, t + i.content + r);
}
const KC = (e) => e === "unset" || e === "undefined" || e === "none";
function QC(e, t) {
  const r = {
    ...af,
    ...e
  }, i = {
    ...qC,
    ...t
  }, a = {
    left: r.left,
    top: r.top,
    width: r.width,
    height: r.height
  };
  let n = r.body;
  [r, i].forEach((m) => {
    const y = [], x = m.hFlip, b = m.vFlip;
    let k = m.rotate;
    x ? b ? k += 2 : (y.push(
      "translate(" + (a.width + a.left).toString() + " " + (0 - a.top).toString() + ")"
    ), y.push("scale(-1 1)"), a.top = a.left = 0) : b && (y.push(
      "translate(" + (0 - a.left).toString() + " " + (a.height + a.top).toString() + ")"
    ), y.push("scale(1 -1)"), a.top = a.left = 0);
    let S;
    switch (k < 0 && (k -= Math.floor(k / 4) * 4), k = k % 4, k) {
      case 1:
        S = a.height / 2 + a.top, y.unshift(
          "rotate(90 " + S.toString() + " " + S.toString() + ")"
        );
        break;
      case 2:
        y.unshift(
          "rotate(180 " + (a.width / 2 + a.left).toString() + " " + (a.height / 2 + a.top).toString() + ")"
        );
        break;
      case 3:
        S = a.width / 2 + a.left, y.unshift(
          "rotate(-90 " + S.toString() + " " + S.toString() + ")"
        );
        break;
    }
    k % 2 === 1 && (a.left !== a.top && (S = a.left, a.left = a.top, a.top = S), a.width !== a.height && (S = a.width, a.width = a.height, a.height = S)), y.length && (n = ZC(
      n,
      '<g transform="' + y.join(" ") + '">',
      "</g>"
    ));
  });
  const o = i.width, s = i.height, l = a.width, c = a.height;
  let h, u;
  o === null ? (u = s === null ? "1em" : s === "auto" ? c : s, h = sl(u, l / c)) : (h = o === "auto" ? l : o, u = s === null ? sl(h, c / l) : s === "auto" ? c : s);
  const f = {}, d = (m, y) => {
    KC(y) || (f[m] = y.toString());
  };
  d("width", h), d("height", u);
  const g = [a.left, a.top, l, c];
  return f.viewBox = g.join(" "), {
    attributes: f,
    viewBox: g,
    body: n
  };
}
const JC = /\sid="(\S+)"/g, tk = "IconifyId" + Date.now().toString(16) + (Math.random() * 16777216 | 0).toString(16);
let ek = 0;
function rk(e, t = tk) {
  const r = [];
  let i;
  for (; i = JC.exec(e); )
    r.push(i[1]);
  if (!r.length)
    return e;
  const a = "suffix" + (Math.random() * 16777216 | Date.now()).toString(16);
  return r.forEach((n) => {
    const o = typeof t == "function" ? t(n) : t + (ek++).toString(), s = n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    e = e.replace(
      // Allowed characters before id: [#;"]
      // Allowed characters after id: [)"], .[a-z]
      new RegExp('([#;"])(' + s + ')([")]|\\.[a-z])', "g"),
      "$1" + o + a + "$3"
    );
  }), e = e.replace(new RegExp(a, "g"), ""), e;
}
function ik(e, t) {
  let r = e.indexOf("xlink:") === -1 ? "" : ' xmlns:xlink="http://www.w3.org/1999/xlink"';
  for (const i in t)
    r += " " + i + '="' + t[i] + '"';
  return '<svg xmlns="http://www.w3.org/2000/svg"' + r + ">" + e + "</svg>";
}
function Xs() {
  return { async: !1, breaks: !1, extensions: null, gfm: !0, hooks: null, pedantic: !1, renderer: null, silent: !1, tokenizer: null, walkTokens: null };
}
var Ke = Xs();
function nf(e) {
  Ke = e;
}
var Zr = { exec: () => null };
function st(e, t = "") {
  let r = typeof e == "string" ? e : e.source, i = { replace: (a, n) => {
    let o = typeof n == "string" ? n : n.source;
    return o = o.replace(Ft.caret, "$1"), r = r.replace(a, o), i;
  }, getRegex: () => new RegExp(r, t) };
  return i;
}
var Ft = { codeRemoveIndent: /^(?: {1,4}| {0,3}\t)/gm, outputLinkReplace: /\\([\[\]])/g, indentCodeCompensation: /^(\s+)(?:```)/, beginningSpace: /^\s+/, endingHash: /#$/, startingSpaceChar: /^ /, endingSpaceChar: / $/, nonSpaceChar: /[^ ]/, newLineCharGlobal: /\n/g, tabCharGlobal: /\t/g, multipleSpaceGlobal: /\s+/g, blankLine: /^[ \t]*$/, doubleBlankLine: /\n[ \t]*\n[ \t]*$/, blockquoteStart: /^ {0,3}>/, blockquoteSetextReplace: /\n {0,3}((?:=+|-+) *)(?=\n|$)/g, blockquoteSetextReplace2: /^ {0,3}>[ \t]?/gm, listReplaceTabs: /^\t+/, listReplaceNesting: /^ {1,4}(?=( {4})*[^ ])/g, listIsTask: /^\[[ xX]\] /, listReplaceTask: /^\[[ xX]\] +/, anyLine: /\n.*\n/, hrefBrackets: /^<(.*)>$/, tableDelimiter: /[:|]/, tableAlignChars: /^\||\| *$/g, tableRowBlankLine: /\n[ \t]*$/, tableAlignRight: /^ *-+: *$/, tableAlignCenter: /^ *:-+: *$/, tableAlignLeft: /^ *:-+ *$/, startATag: /^<a /i, endATag: /^<\/a>/i, startPreScriptTag: /^<(pre|code|kbd|script)(\s|>)/i, endPreScriptTag: /^<\/(pre|code|kbd|script)(\s|>)/i, startAngleBracket: /^</, endAngleBracket: />$/, pedanticHrefTitle: /^([^'"]*[^\s])\s+(['"])(.*)\2/, unicodeAlphaNumeric: /[\p{L}\p{N}]/u, escapeTest: /[&<>"']/, escapeReplace: /[&<>"']/g, escapeTestNoEncode: /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/, escapeReplaceNoEncode: /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/g, unescapeTest: /&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/ig, caret: /(^|[^\[])\^/g, percentDecode: /%25/g, findPipe: /\|/g, splitPipe: / \|/, slashPipe: /\\\|/g, carriageReturn: /\r\n|\r/g, spaceLine: /^ +$/gm, notSpaceStart: /^\S*/, endingNewline: /\n$/, listItemRegex: (e) => new RegExp(`^( {0,3}${e})((?:[	 ][^\\n]*)?(?:\\n|$))`), nextBulletRegex: (e) => new RegExp(`^ {0,${Math.min(3, e - 1)}}(?:[*+-]|\\d{1,9}[.)])((?:[ 	][^\\n]*)?(?:\\n|$))`), hrRegex: (e) => new RegExp(`^ {0,${Math.min(3, e - 1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`), fencesBeginRegex: (e) => new RegExp(`^ {0,${Math.min(3, e - 1)}}(?:\`\`\`|~~~)`), headingBeginRegex: (e) => new RegExp(`^ {0,${Math.min(3, e - 1)}}#`), htmlBeginRegex: (e) => new RegExp(`^ {0,${Math.min(3, e - 1)}}<(?:[a-z].*>|!--)`, "i") }, ak = /^(?:[ \t]*(?:\n|$))+/, nk = /^((?: {4}| {0,3}\t)[^\n]+(?:\n(?:[ \t]*(?:\n|$))*)?)+/, sk = /^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/, mi = /^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/, ok = /^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/, Vs = /(?:[*+-]|\d{1,9}[.)])/, sf = /^(?!bull |blockCode|fences|blockquote|heading|html|table)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html|table))+?)\n {0,3}(=+|-+) *(?:\n+|$)/, of = st(sf).replace(/bull/g, Vs).replace(/blockCode/g, /(?: {4}| {0,3}\t)/).replace(/fences/g, / {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g, / {0,3}>/).replace(/heading/g, / {0,3}#{1,6}/).replace(/html/g, / {0,3}<[^\n>]+>\n/).replace(/\|table/g, "").getRegex(), lk = st(sf).replace(/bull/g, Vs).replace(/blockCode/g, /(?: {4}| {0,3}\t)/).replace(/fences/g, / {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g, / {0,3}>/).replace(/heading/g, / {0,3}#{1,6}/).replace(/html/g, / {0,3}<[^\n>]+>\n/).replace(/table/g, / {0,3}\|?(?:[:\- ]*\|)+[\:\- ]*\n/).getRegex(), Zs = /^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/, ck = /^[^\n]+/, Ks = /(?!\s*\])(?:\\.|[^\[\]\\])+/, hk = st(/^ {0,3}\[(label)\]: *(?:\n[ \t]*)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n[ \t]*)?| *\n[ \t]*)(title))? *(?:\n+|$)/).replace("label", Ks).replace("title", /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex(), uk = st(/^( {0,3}bull)([ \t][^\n]+?)?(?:\n|$)/).replace(/bull/g, Vs).getRegex(), Wa = "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul", Qs = /<!--(?:-?>|[\s\S]*?(?:-->|$))/, fk = st("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$))", "i").replace("comment", Qs).replace("tag", Wa).replace("attribute", / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex(), lf = st(Zs).replace("hr", mi).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("|table", "").replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", Wa).getRegex(), dk = st(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph", lf).getRegex(), Js = { blockquote: dk, code: nk, def: hk, fences: sk, heading: ok, hr: mi, html: fk, lheading: of, list: uk, newline: ak, paragraph: lf, table: Zr, text: ck }, ol = st("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr", mi).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("blockquote", " {0,3}>").replace("code", "(?: {4}| {0,3}	)[^\\n]").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", Wa).getRegex(), pk = { ...Js, lheading: lk, table: ol, paragraph: st(Zs).replace("hr", mi).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("table", ol).replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", Wa).getRegex() }, gk = { ...Js, html: st(`^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:"[^"]*"|'[^']*'|\\s[^'"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))`).replace("comment", Qs).replace(/tag/g, "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(), def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/, heading: /^(#{1,6})(.*)(?:\n+|$)/, fences: Zr, lheading: /^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/, paragraph: st(Zs).replace("hr", mi).replace("heading", ` *#{1,6} *[^
]`).replace("lheading", of).replace("|table", "").replace("blockquote", " {0,3}>").replace("|fences", "").replace("|list", "").replace("|html", "").replace("|tag", "").getRegex() }, mk = /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/, yk = /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/, cf = /^( {2,}|\\)\n(?!\s*$)/, xk = /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/, Ha = /[\p{P}\p{S}]/u, to = /[\s\p{P}\p{S}]/u, hf = /[^\s\p{P}\p{S}]/u, bk = st(/^((?![*_])punctSpace)/, "u").replace(/punctSpace/g, to).getRegex(), uf = /(?!~)[\p{P}\p{S}]/u, Ck = /(?!~)[\s\p{P}\p{S}]/u, kk = /(?:[^\s\p{P}\p{S}]|~)/u, wk = /\[[^[\]]*?\]\((?:\\.|[^\\\(\)]|\((?:\\.|[^\\\(\)])*\))*\)|`[^`]*?`|<(?! )[^<>]*?>/g, ff = /^(?:\*+(?:((?!\*)punct)|[^\s*]))|^_+(?:((?!_)punct)|([^\s_]))/, _k = st(ff, "u").replace(/punct/g, Ha).getRegex(), vk = st(ff, "u").replace(/punct/g, uf).getRegex(), df = "^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)punct(\\*+)(?=[\\s]|$)|notPunctSpace(\\*+)(?!\\*)(?=punctSpace|$)|(?!\\*)punctSpace(\\*+)(?=notPunctSpace)|[\\s](\\*+)(?!\\*)(?=punct)|(?!\\*)punct(\\*+)(?!\\*)(?=punct)|notPunctSpace(\\*+)(?=notPunctSpace)", Sk = st(df, "gu").replace(/notPunctSpace/g, hf).replace(/punctSpace/g, to).replace(/punct/g, Ha).getRegex(), Tk = st(df, "gu").replace(/notPunctSpace/g, kk).replace(/punctSpace/g, Ck).replace(/punct/g, uf).getRegex(), Bk = st("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)punct(_+)(?=[\\s]|$)|notPunctSpace(_+)(?!_)(?=punctSpace|$)|(?!_)punctSpace(_+)(?=notPunctSpace)|[\\s](_+)(?!_)(?=punct)|(?!_)punct(_+)(?!_)(?=punct)", "gu").replace(/notPunctSpace/g, hf).replace(/punctSpace/g, to).replace(/punct/g, Ha).getRegex(), Lk = st(/\\(punct)/, "gu").replace(/punct/g, Ha).getRegex(), Mk = st(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme", /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email", /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex(), $k = st(Qs).replace("(?:-->|$)", "-->").getRegex(), Ak = st("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment", $k).replace("attribute", /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex(), pa = /(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/, Fk = st(/^!?\[(label)\]\(\s*(href)(?:(?:[ \t]*(?:\n[ \t]*)?)(title))?\s*\)/).replace("label", pa).replace("href", /<(?:\\.|[^\n<>\\])+>|[^ \t\n\x00-\x1f]*/).replace("title", /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex(), pf = st(/^!?\[(label)\]\[(ref)\]/).replace("label", pa).replace("ref", Ks).getRegex(), gf = st(/^!?\[(ref)\](?:\[\])?/).replace("ref", Ks).getRegex(), Ek = st("reflink|nolink(?!\\()", "g").replace("reflink", pf).replace("nolink", gf).getRegex(), eo = { _backpedal: Zr, anyPunctuation: Lk, autolink: Mk, blockSkip: wk, br: cf, code: yk, del: Zr, emStrongLDelim: _k, emStrongRDelimAst: Sk, emStrongRDelimUnd: Bk, escape: mk, link: Fk, nolink: gf, punctuation: bk, reflink: pf, reflinkSearch: Ek, tag: Ak, text: xk, url: Zr }, Ok = { ...eo, link: st(/^!?\[(label)\]\((.*?)\)/).replace("label", pa).getRegex(), reflink: st(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label", pa).getRegex() }, Gn = { ...eo, emStrongRDelimAst: Tk, emStrongLDelim: vk, url: st(/^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/, "i").replace("email", /[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(), _backpedal: /(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/, del: /^(~~?)(?=[^\s~])((?:\\.|[^\\])*?(?:\\.|[^\s~\\]))\1(?=[^~]|$)/, text: /^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/ }, Dk = { ...Gn, br: st(cf).replace("{2,}", "*").getRegex(), text: st(Gn.text).replace("\\b_", "\\b_| {2,}\\n").replace(/\{2,\}/g, "*").getRegex() }, vi = { normal: Js, gfm: pk, pedantic: gk }, Er = { normal: eo, gfm: Gn, breaks: Dk, pedantic: Ok }, Rk = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }, ll = (e) => Rk[e];
function Zt(e, t) {
  if (t) {
    if (Ft.escapeTest.test(e)) return e.replace(Ft.escapeReplace, ll);
  } else if (Ft.escapeTestNoEncode.test(e)) return e.replace(Ft.escapeReplaceNoEncode, ll);
  return e;
}
function cl(e) {
  try {
    e = encodeURI(e).replace(Ft.percentDecode, "%");
  } catch {
    return null;
  }
  return e;
}
function hl(e, t) {
  var n;
  let r = e.replace(Ft.findPipe, (o, s, l) => {
    let c = !1, h = s;
    for (; --h >= 0 && l[h] === "\\"; ) c = !c;
    return c ? "|" : " |";
  }), i = r.split(Ft.splitPipe), a = 0;
  if (i[0].trim() || i.shift(), i.length > 0 && !((n = i.at(-1)) != null && n.trim()) && i.pop(), t) if (i.length > t) i.splice(t);
  else for (; i.length < t; ) i.push("");
  for (; a < i.length; a++) i[a] = i[a].trim().replace(Ft.slashPipe, "|");
  return i;
}
function Or(e, t, r) {
  let i = e.length;
  if (i === 0) return "";
  let a = 0;
  for (; a < i && e.charAt(i - a - 1) === t; )
    a++;
  return e.slice(0, i - a);
}
function Pk(e, t) {
  if (e.indexOf(t[1]) === -1) return -1;
  let r = 0;
  for (let i = 0; i < e.length; i++) if (e[i] === "\\") i++;
  else if (e[i] === t[0]) r++;
  else if (e[i] === t[1] && (r--, r < 0)) return i;
  return r > 0 ? -2 : -1;
}
function ul(e, t, r, i, a) {
  let n = t.href, o = t.title || null, s = e[1].replace(a.other.outputLinkReplace, "$1");
  i.state.inLink = !0;
  let l = { type: e[0].charAt(0) === "!" ? "image" : "link", raw: r, href: n, title: o, text: s, tokens: i.inlineTokens(s) };
  return i.state.inLink = !1, l;
}
function Ik(e, t, r) {
  let i = e.match(r.other.indentCodeCompensation);
  if (i === null) return t;
  let a = i[1];
  return t.split(`
`).map((n) => {
    let o = n.match(r.other.beginningSpace);
    if (o === null) return n;
    let [s] = o;
    return s.length >= a.length ? n.slice(a.length) : n;
  }).join(`
`);
}
var ga = class {
  constructor(t) {
    ot(this, "options");
    ot(this, "rules");
    ot(this, "lexer");
    this.options = t || Ke;
  }
  space(t) {
    let r = this.rules.block.newline.exec(t);
    if (r && r[0].length > 0) return { type: "space", raw: r[0] };
  }
  code(t) {
    let r = this.rules.block.code.exec(t);
    if (r) {
      let i = r[0].replace(this.rules.other.codeRemoveIndent, "");
      return { type: "code", raw: r[0], codeBlockStyle: "indented", text: this.options.pedantic ? i : Or(i, `
`) };
    }
  }
  fences(t) {
    let r = this.rules.block.fences.exec(t);
    if (r) {
      let i = r[0], a = Ik(i, r[3] || "", this.rules);
      return { type: "code", raw: i, lang: r[2] ? r[2].trim().replace(this.rules.inline.anyPunctuation, "$1") : r[2], text: a };
    }
  }
  heading(t) {
    let r = this.rules.block.heading.exec(t);
    if (r) {
      let i = r[2].trim();
      if (this.rules.other.endingHash.test(i)) {
        let a = Or(i, "#");
        (this.options.pedantic || !a || this.rules.other.endingSpaceChar.test(a)) && (i = a.trim());
      }
      return { type: "heading", raw: r[0], depth: r[1].length, text: i, tokens: this.lexer.inline(i) };
    }
  }
  hr(t) {
    let r = this.rules.block.hr.exec(t);
    if (r) return { type: "hr", raw: Or(r[0], `
`) };
  }
  blockquote(t) {
    let r = this.rules.block.blockquote.exec(t);
    if (r) {
      let i = Or(r[0], `
`).split(`
`), a = "", n = "", o = [];
      for (; i.length > 0; ) {
        let s = !1, l = [], c;
        for (c = 0; c < i.length; c++) if (this.rules.other.blockquoteStart.test(i[c])) l.push(i[c]), s = !0;
        else if (!s) l.push(i[c]);
        else break;
        i = i.slice(c);
        let h = l.join(`
`), u = h.replace(this.rules.other.blockquoteSetextReplace, `
    $1`).replace(this.rules.other.blockquoteSetextReplace2, "");
        a = a ? `${a}
${h}` : h, n = n ? `${n}
${u}` : u;
        let f = this.lexer.state.top;
        if (this.lexer.state.top = !0, this.lexer.blockTokens(u, o, !0), this.lexer.state.top = f, i.length === 0) break;
        let d = o.at(-1);
        if ((d == null ? void 0 : d.type) === "code") break;
        if ((d == null ? void 0 : d.type) === "blockquote") {
          let g = d, m = g.raw + `
` + i.join(`
`), y = this.blockquote(m);
          o[o.length - 1] = y, a = a.substring(0, a.length - g.raw.length) + y.raw, n = n.substring(0, n.length - g.text.length) + y.text;
          break;
        } else if ((d == null ? void 0 : d.type) === "list") {
          let g = d, m = g.raw + `
` + i.join(`
`), y = this.list(m);
          o[o.length - 1] = y, a = a.substring(0, a.length - d.raw.length) + y.raw, n = n.substring(0, n.length - g.raw.length) + y.raw, i = m.substring(o.at(-1).raw.length).split(`
`);
          continue;
        }
      }
      return { type: "blockquote", raw: a, tokens: o, text: n };
    }
  }
  list(t) {
    let r = this.rules.block.list.exec(t);
    if (r) {
      let i = r[1].trim(), a = i.length > 1, n = { type: "list", raw: "", ordered: a, start: a ? +i.slice(0, -1) : "", loose: !1, items: [] };
      i = a ? `\\d{1,9}\\${i.slice(-1)}` : `\\${i}`, this.options.pedantic && (i = a ? i : "[*+-]");
      let o = this.rules.other.listItemRegex(i), s = !1;
      for (; t; ) {
        let c = !1, h = "", u = "";
        if (!(r = o.exec(t)) || this.rules.block.hr.test(t)) break;
        h = r[0], t = t.substring(h.length);
        let f = r[2].split(`
`, 1)[0].replace(this.rules.other.listReplaceTabs, (b) => " ".repeat(3 * b.length)), d = t.split(`
`, 1)[0], g = !f.trim(), m = 0;
        if (this.options.pedantic ? (m = 2, u = f.trimStart()) : g ? m = r[1].length + 1 : (m = r[2].search(this.rules.other.nonSpaceChar), m = m > 4 ? 1 : m, u = f.slice(m), m += r[1].length), g && this.rules.other.blankLine.test(d) && (h += d + `
`, t = t.substring(d.length + 1), c = !0), !c) {
          let b = this.rules.other.nextBulletRegex(m), k = this.rules.other.hrRegex(m), S = this.rules.other.fencesBeginRegex(m), w = this.rules.other.headingBeginRegex(m), C = this.rules.other.htmlBeginRegex(m);
          for (; t; ) {
            let _ = t.split(`
`, 1)[0], E;
            if (d = _, this.options.pedantic ? (d = d.replace(this.rules.other.listReplaceNesting, "  "), E = d) : E = d.replace(this.rules.other.tabCharGlobal, "    "), S.test(d) || w.test(d) || C.test(d) || b.test(d) || k.test(d)) break;
            if (E.search(this.rules.other.nonSpaceChar) >= m || !d.trim()) u += `
` + E.slice(m);
            else {
              if (g || f.replace(this.rules.other.tabCharGlobal, "    ").search(this.rules.other.nonSpaceChar) >= 4 || S.test(f) || w.test(f) || k.test(f)) break;
              u += `
` + d;
            }
            !g && !d.trim() && (g = !0), h += _ + `
`, t = t.substring(_.length + 1), f = E.slice(m);
          }
        }
        n.loose || (s ? n.loose = !0 : this.rules.other.doubleBlankLine.test(h) && (s = !0));
        let y = null, x;
        this.options.gfm && (y = this.rules.other.listIsTask.exec(u), y && (x = y[0] !== "[ ] ", u = u.replace(this.rules.other.listReplaceTask, ""))), n.items.push({ type: "list_item", raw: h, task: !!y, checked: x, loose: !1, text: u, tokens: [] }), n.raw += h;
      }
      let l = n.items.at(-1);
      if (l) l.raw = l.raw.trimEnd(), l.text = l.text.trimEnd();
      else return;
      n.raw = n.raw.trimEnd();
      for (let c = 0; c < n.items.length; c++) if (this.lexer.state.top = !1, n.items[c].tokens = this.lexer.blockTokens(n.items[c].text, []), !n.loose) {
        let h = n.items[c].tokens.filter((f) => f.type === "space"), u = h.length > 0 && h.some((f) => this.rules.other.anyLine.test(f.raw));
        n.loose = u;
      }
      if (n.loose) for (let c = 0; c < n.items.length; c++) n.items[c].loose = !0;
      return n;
    }
  }
  html(t) {
    let r = this.rules.block.html.exec(t);
    if (r) return { type: "html", block: !0, raw: r[0], pre: r[1] === "pre" || r[1] === "script" || r[1] === "style", text: r[0] };
  }
  def(t) {
    let r = this.rules.block.def.exec(t);
    if (r) {
      let i = r[1].toLowerCase().replace(this.rules.other.multipleSpaceGlobal, " "), a = r[2] ? r[2].replace(this.rules.other.hrefBrackets, "$1").replace(this.rules.inline.anyPunctuation, "$1") : "", n = r[3] ? r[3].substring(1, r[3].length - 1).replace(this.rules.inline.anyPunctuation, "$1") : r[3];
      return { type: "def", tag: i, raw: r[0], href: a, title: n };
    }
  }
  table(t) {
    var s;
    let r = this.rules.block.table.exec(t);
    if (!r || !this.rules.other.tableDelimiter.test(r[2])) return;
    let i = hl(r[1]), a = r[2].replace(this.rules.other.tableAlignChars, "").split("|"), n = (s = r[3]) != null && s.trim() ? r[3].replace(this.rules.other.tableRowBlankLine, "").split(`
`) : [], o = { type: "table", raw: r[0], header: [], align: [], rows: [] };
    if (i.length === a.length) {
      for (let l of a) this.rules.other.tableAlignRight.test(l) ? o.align.push("right") : this.rules.other.tableAlignCenter.test(l) ? o.align.push("center") : this.rules.other.tableAlignLeft.test(l) ? o.align.push("left") : o.align.push(null);
      for (let l = 0; l < i.length; l++) o.header.push({ text: i[l], tokens: this.lexer.inline(i[l]), header: !0, align: o.align[l] });
      for (let l of n) o.rows.push(hl(l, o.header.length).map((c, h) => ({ text: c, tokens: this.lexer.inline(c), header: !1, align: o.align[h] })));
      return o;
    }
  }
  lheading(t) {
    let r = this.rules.block.lheading.exec(t);
    if (r) return { type: "heading", raw: r[0], depth: r[2].charAt(0) === "=" ? 1 : 2, text: r[1], tokens: this.lexer.inline(r[1]) };
  }
  paragraph(t) {
    let r = this.rules.block.paragraph.exec(t);
    if (r) {
      let i = r[1].charAt(r[1].length - 1) === `
` ? r[1].slice(0, -1) : r[1];
      return { type: "paragraph", raw: r[0], text: i, tokens: this.lexer.inline(i) };
    }
  }
  text(t) {
    let r = this.rules.block.text.exec(t);
    if (r) return { type: "text", raw: r[0], text: r[0], tokens: this.lexer.inline(r[0]) };
  }
  escape(t) {
    let r = this.rules.inline.escape.exec(t);
    if (r) return { type: "escape", raw: r[0], text: r[1] };
  }
  tag(t) {
    let r = this.rules.inline.tag.exec(t);
    if (r) return !this.lexer.state.inLink && this.rules.other.startATag.test(r[0]) ? this.lexer.state.inLink = !0 : this.lexer.state.inLink && this.rules.other.endATag.test(r[0]) && (this.lexer.state.inLink = !1), !this.lexer.state.inRawBlock && this.rules.other.startPreScriptTag.test(r[0]) ? this.lexer.state.inRawBlock = !0 : this.lexer.state.inRawBlock && this.rules.other.endPreScriptTag.test(r[0]) && (this.lexer.state.inRawBlock = !1), { type: "html", raw: r[0], inLink: this.lexer.state.inLink, inRawBlock: this.lexer.state.inRawBlock, block: !1, text: r[0] };
  }
  link(t) {
    let r = this.rules.inline.link.exec(t);
    if (r) {
      let i = r[2].trim();
      if (!this.options.pedantic && this.rules.other.startAngleBracket.test(i)) {
        if (!this.rules.other.endAngleBracket.test(i)) return;
        let o = Or(i.slice(0, -1), "\\");
        if ((i.length - o.length) % 2 === 0) return;
      } else {
        let o = Pk(r[2], "()");
        if (o === -2) return;
        if (o > -1) {
          let s = (r[0].indexOf("!") === 0 ? 5 : 4) + r[1].length + o;
          r[2] = r[2].substring(0, o), r[0] = r[0].substring(0, s).trim(), r[3] = "";
        }
      }
      let a = r[2], n = "";
      if (this.options.pedantic) {
        let o = this.rules.other.pedanticHrefTitle.exec(a);
        o && (a = o[1], n = o[3]);
      } else n = r[3] ? r[3].slice(1, -1) : "";
      return a = a.trim(), this.rules.other.startAngleBracket.test(a) && (this.options.pedantic && !this.rules.other.endAngleBracket.test(i) ? a = a.slice(1) : a = a.slice(1, -1)), ul(r, { href: a && a.replace(this.rules.inline.anyPunctuation, "$1"), title: n && n.replace(this.rules.inline.anyPunctuation, "$1") }, r[0], this.lexer, this.rules);
    }
  }
  reflink(t, r) {
    let i;
    if ((i = this.rules.inline.reflink.exec(t)) || (i = this.rules.inline.nolink.exec(t))) {
      let a = (i[2] || i[1]).replace(this.rules.other.multipleSpaceGlobal, " "), n = r[a.toLowerCase()];
      if (!n) {
        let o = i[0].charAt(0);
        return { type: "text", raw: o, text: o };
      }
      return ul(i, n, i[0], this.lexer, this.rules);
    }
  }
  emStrong(t, r, i = "") {
    let a = this.rules.inline.emStrongLDelim.exec(t);
    if (!(!a || a[3] && i.match(this.rules.other.unicodeAlphaNumeric)) && (!(a[1] || a[2]) || !i || this.rules.inline.punctuation.exec(i))) {
      let n = [...a[0]].length - 1, o, s, l = n, c = 0, h = a[0][0] === "*" ? this.rules.inline.emStrongRDelimAst : this.rules.inline.emStrongRDelimUnd;
      for (h.lastIndex = 0, r = r.slice(-1 * t.length + n); (a = h.exec(r)) != null; ) {
        if (o = a[1] || a[2] || a[3] || a[4] || a[5] || a[6], !o) continue;
        if (s = [...o].length, a[3] || a[4]) {
          l += s;
          continue;
        } else if ((a[5] || a[6]) && n % 3 && !((n + s) % 3)) {
          c += s;
          continue;
        }
        if (l -= s, l > 0) continue;
        s = Math.min(s, s + l + c);
        let u = [...a[0]][0].length, f = t.slice(0, n + a.index + u + s);
        if (Math.min(n, s) % 2) {
          let g = f.slice(1, -1);
          return { type: "em", raw: f, text: g, tokens: this.lexer.inlineTokens(g) };
        }
        let d = f.slice(2, -2);
        return { type: "strong", raw: f, text: d, tokens: this.lexer.inlineTokens(d) };
      }
    }
  }
  codespan(t) {
    let r = this.rules.inline.code.exec(t);
    if (r) {
      let i = r[2].replace(this.rules.other.newLineCharGlobal, " "), a = this.rules.other.nonSpaceChar.test(i), n = this.rules.other.startingSpaceChar.test(i) && this.rules.other.endingSpaceChar.test(i);
      return a && n && (i = i.substring(1, i.length - 1)), { type: "codespan", raw: r[0], text: i };
    }
  }
  br(t) {
    let r = this.rules.inline.br.exec(t);
    if (r) return { type: "br", raw: r[0] };
  }
  del(t) {
    let r = this.rules.inline.del.exec(t);
    if (r) return { type: "del", raw: r[0], text: r[2], tokens: this.lexer.inlineTokens(r[2]) };
  }
  autolink(t) {
    let r = this.rules.inline.autolink.exec(t);
    if (r) {
      let i, a;
      return r[2] === "@" ? (i = r[1], a = "mailto:" + i) : (i = r[1], a = i), { type: "link", raw: r[0], text: i, href: a, tokens: [{ type: "text", raw: i, text: i }] };
    }
  }
  url(t) {
    var i;
    let r;
    if (r = this.rules.inline.url.exec(t)) {
      let a, n;
      if (r[2] === "@") a = r[0], n = "mailto:" + a;
      else {
        let o;
        do
          o = r[0], r[0] = ((i = this.rules.inline._backpedal.exec(r[0])) == null ? void 0 : i[0]) ?? "";
        while (o !== r[0]);
        a = r[0], r[1] === "www." ? n = "http://" + r[0] : n = r[0];
      }
      return { type: "link", raw: r[0], text: a, href: n, tokens: [{ type: "text", raw: a, text: a }] };
    }
  }
  inlineText(t) {
    let r = this.rules.inline.text.exec(t);
    if (r) {
      let i = this.lexer.state.inRawBlock;
      return { type: "text", raw: r[0], text: r[0], escaped: i };
    }
  }
}, ce = class Un {
  constructor(t) {
    ot(this, "tokens");
    ot(this, "options");
    ot(this, "state");
    ot(this, "tokenizer");
    ot(this, "inlineQueue");
    this.tokens = [], this.tokens.links = /* @__PURE__ */ Object.create(null), this.options = t || Ke, this.options.tokenizer = this.options.tokenizer || new ga(), this.tokenizer = this.options.tokenizer, this.tokenizer.options = this.options, this.tokenizer.lexer = this, this.inlineQueue = [], this.state = { inLink: !1, inRawBlock: !1, top: !0 };
    let r = { other: Ft, block: vi.normal, inline: Er.normal };
    this.options.pedantic ? (r.block = vi.pedantic, r.inline = Er.pedantic) : this.options.gfm && (r.block = vi.gfm, this.options.breaks ? r.inline = Er.breaks : r.inline = Er.gfm), this.tokenizer.rules = r;
  }
  static get rules() {
    return { block: vi, inline: Er };
  }
  static lex(t, r) {
    return new Un(r).lex(t);
  }
  static lexInline(t, r) {
    return new Un(r).inlineTokens(t);
  }
  lex(t) {
    t = t.replace(Ft.carriageReturn, `
`), this.blockTokens(t, this.tokens);
    for (let r = 0; r < this.inlineQueue.length; r++) {
      let i = this.inlineQueue[r];
      this.inlineTokens(i.src, i.tokens);
    }
    return this.inlineQueue = [], this.tokens;
  }
  blockTokens(t, r = [], i = !1) {
    var a, n, o;
    for (this.options.pedantic && (t = t.replace(Ft.tabCharGlobal, "    ").replace(Ft.spaceLine, "")); t; ) {
      let s;
      if ((n = (a = this.options.extensions) == null ? void 0 : a.block) != null && n.some((c) => (s = c.call({ lexer: this }, t, r)) ? (t = t.substring(s.raw.length), r.push(s), !0) : !1)) continue;
      if (s = this.tokenizer.space(t)) {
        t = t.substring(s.raw.length);
        let c = r.at(-1);
        s.raw.length === 1 && c !== void 0 ? c.raw += `
` : r.push(s);
        continue;
      }
      if (s = this.tokenizer.code(t)) {
        t = t.substring(s.raw.length);
        let c = r.at(-1);
        (c == null ? void 0 : c.type) === "paragraph" || (c == null ? void 0 : c.type) === "text" ? (c.raw += (c.raw.endsWith(`
`) ? "" : `
`) + s.raw, c.text += `
` + s.text, this.inlineQueue.at(-1).src = c.text) : r.push(s);
        continue;
      }
      if (s = this.tokenizer.fences(t)) {
        t = t.substring(s.raw.length), r.push(s);
        continue;
      }
      if (s = this.tokenizer.heading(t)) {
        t = t.substring(s.raw.length), r.push(s);
        continue;
      }
      if (s = this.tokenizer.hr(t)) {
        t = t.substring(s.raw.length), r.push(s);
        continue;
      }
      if (s = this.tokenizer.blockquote(t)) {
        t = t.substring(s.raw.length), r.push(s);
        continue;
      }
      if (s = this.tokenizer.list(t)) {
        t = t.substring(s.raw.length), r.push(s);
        continue;
      }
      if (s = this.tokenizer.html(t)) {
        t = t.substring(s.raw.length), r.push(s);
        continue;
      }
      if (s = this.tokenizer.def(t)) {
        t = t.substring(s.raw.length);
        let c = r.at(-1);
        (c == null ? void 0 : c.type) === "paragraph" || (c == null ? void 0 : c.type) === "text" ? (c.raw += (c.raw.endsWith(`
`) ? "" : `
`) + s.raw, c.text += `
` + s.raw, this.inlineQueue.at(-1).src = c.text) : this.tokens.links[s.tag] || (this.tokens.links[s.tag] = { href: s.href, title: s.title });
        continue;
      }
      if (s = this.tokenizer.table(t)) {
        t = t.substring(s.raw.length), r.push(s);
        continue;
      }
      if (s = this.tokenizer.lheading(t)) {
        t = t.substring(s.raw.length), r.push(s);
        continue;
      }
      let l = t;
      if ((o = this.options.extensions) != null && o.startBlock) {
        let c = 1 / 0, h = t.slice(1), u;
        this.options.extensions.startBlock.forEach((f) => {
          u = f.call({ lexer: this }, h), typeof u == "number" && u >= 0 && (c = Math.min(c, u));
        }), c < 1 / 0 && c >= 0 && (l = t.substring(0, c + 1));
      }
      if (this.state.top && (s = this.tokenizer.paragraph(l))) {
        let c = r.at(-1);
        i && (c == null ? void 0 : c.type) === "paragraph" ? (c.raw += (c.raw.endsWith(`
`) ? "" : `
`) + s.raw, c.text += `
` + s.text, this.inlineQueue.pop(), this.inlineQueue.at(-1).src = c.text) : r.push(s), i = l.length !== t.length, t = t.substring(s.raw.length);
        continue;
      }
      if (s = this.tokenizer.text(t)) {
        t = t.substring(s.raw.length);
        let c = r.at(-1);
        (c == null ? void 0 : c.type) === "text" ? (c.raw += (c.raw.endsWith(`
`) ? "" : `
`) + s.raw, c.text += `
` + s.text, this.inlineQueue.pop(), this.inlineQueue.at(-1).src = c.text) : r.push(s);
        continue;
      }
      if (t) {
        let c = "Infinite loop on byte: " + t.charCodeAt(0);
        if (this.options.silent) {
          console.error(c);
          break;
        } else throw new Error(c);
      }
    }
    return this.state.top = !0, r;
  }
  inline(t, r = []) {
    return this.inlineQueue.push({ src: t, tokens: r }), r;
  }
  inlineTokens(t, r = []) {
    var s, l, c;
    let i = t, a = null;
    if (this.tokens.links) {
      let h = Object.keys(this.tokens.links);
      if (h.length > 0) for (; (a = this.tokenizer.rules.inline.reflinkSearch.exec(i)) != null; ) h.includes(a[0].slice(a[0].lastIndexOf("[") + 1, -1)) && (i = i.slice(0, a.index) + "[" + "a".repeat(a[0].length - 2) + "]" + i.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex));
    }
    for (; (a = this.tokenizer.rules.inline.anyPunctuation.exec(i)) != null; ) i = i.slice(0, a.index) + "++" + i.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);
    for (; (a = this.tokenizer.rules.inline.blockSkip.exec(i)) != null; ) i = i.slice(0, a.index) + "[" + "a".repeat(a[0].length - 2) + "]" + i.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);
    let n = !1, o = "";
    for (; t; ) {
      n || (o = ""), n = !1;
      let h;
      if ((l = (s = this.options.extensions) == null ? void 0 : s.inline) != null && l.some((f) => (h = f.call({ lexer: this }, t, r)) ? (t = t.substring(h.raw.length), r.push(h), !0) : !1)) continue;
      if (h = this.tokenizer.escape(t)) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      if (h = this.tokenizer.tag(t)) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      if (h = this.tokenizer.link(t)) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      if (h = this.tokenizer.reflink(t, this.tokens.links)) {
        t = t.substring(h.raw.length);
        let f = r.at(-1);
        h.type === "text" && (f == null ? void 0 : f.type) === "text" ? (f.raw += h.raw, f.text += h.text) : r.push(h);
        continue;
      }
      if (h = this.tokenizer.emStrong(t, i, o)) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      if (h = this.tokenizer.codespan(t)) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      if (h = this.tokenizer.br(t)) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      if (h = this.tokenizer.del(t)) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      if (h = this.tokenizer.autolink(t)) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      if (!this.state.inLink && (h = this.tokenizer.url(t))) {
        t = t.substring(h.raw.length), r.push(h);
        continue;
      }
      let u = t;
      if ((c = this.options.extensions) != null && c.startInline) {
        let f = 1 / 0, d = t.slice(1), g;
        this.options.extensions.startInline.forEach((m) => {
          g = m.call({ lexer: this }, d), typeof g == "number" && g >= 0 && (f = Math.min(f, g));
        }), f < 1 / 0 && f >= 0 && (u = t.substring(0, f + 1));
      }
      if (h = this.tokenizer.inlineText(u)) {
        t = t.substring(h.raw.length), h.raw.slice(-1) !== "_" && (o = h.raw.slice(-1)), n = !0;
        let f = r.at(-1);
        (f == null ? void 0 : f.type) === "text" ? (f.raw += h.raw, f.text += h.text) : r.push(h);
        continue;
      }
      if (t) {
        let f = "Infinite loop on byte: " + t.charCodeAt(0);
        if (this.options.silent) {
          console.error(f);
          break;
        } else throw new Error(f);
      }
    }
    return r;
  }
}, ma = class {
  constructor(t) {
    ot(this, "options");
    ot(this, "parser");
    this.options = t || Ke;
  }
  space(t) {
    return "";
  }
  code({ text: t, lang: r, escaped: i }) {
    var o;
    let a = (o = (r || "").match(Ft.notSpaceStart)) == null ? void 0 : o[0], n = t.replace(Ft.endingNewline, "") + `
`;
    return a ? '<pre><code class="language-' + Zt(a) + '">' + (i ? n : Zt(n, !0)) + `</code></pre>
` : "<pre><code>" + (i ? n : Zt(n, !0)) + `</code></pre>
`;
  }
  blockquote({ tokens: t }) {
    return `<blockquote>
${this.parser.parse(t)}</blockquote>
`;
  }
  html({ text: t }) {
    return t;
  }
  heading({ tokens: t, depth: r }) {
    return `<h${r}>${this.parser.parseInline(t)}</h${r}>
`;
  }
  hr(t) {
    return `<hr>
`;
  }
  list(t) {
    let r = t.ordered, i = t.start, a = "";
    for (let s = 0; s < t.items.length; s++) {
      let l = t.items[s];
      a += this.listitem(l);
    }
    let n = r ? "ol" : "ul", o = r && i !== 1 ? ' start="' + i + '"' : "";
    return "<" + n + o + `>
` + a + "</" + n + `>
`;
  }
  listitem(t) {
    var i;
    let r = "";
    if (t.task) {
      let a = this.checkbox({ checked: !!t.checked });
      t.loose ? ((i = t.tokens[0]) == null ? void 0 : i.type) === "paragraph" ? (t.tokens[0].text = a + " " + t.tokens[0].text, t.tokens[0].tokens && t.tokens[0].tokens.length > 0 && t.tokens[0].tokens[0].type === "text" && (t.tokens[0].tokens[0].text = a + " " + Zt(t.tokens[0].tokens[0].text), t.tokens[0].tokens[0].escaped = !0)) : t.tokens.unshift({ type: "text", raw: a + " ", text: a + " ", escaped: !0 }) : r += a + " ";
    }
    return r += this.parser.parse(t.tokens, !!t.loose), `<li>${r}</li>
`;
  }
  checkbox({ checked: t }) {
    return "<input " + (t ? 'checked="" ' : "") + 'disabled="" type="checkbox">';
  }
  paragraph({ tokens: t }) {
    return `<p>${this.parser.parseInline(t)}</p>
`;
  }
  table(t) {
    let r = "", i = "";
    for (let n = 0; n < t.header.length; n++) i += this.tablecell(t.header[n]);
    r += this.tablerow({ text: i });
    let a = "";
    for (let n = 0; n < t.rows.length; n++) {
      let o = t.rows[n];
      i = "";
      for (let s = 0; s < o.length; s++) i += this.tablecell(o[s]);
      a += this.tablerow({ text: i });
    }
    return a && (a = `<tbody>${a}</tbody>`), `<table>
<thead>
` + r + `</thead>
` + a + `</table>
`;
  }
  tablerow({ text: t }) {
    return `<tr>
${t}</tr>
`;
  }
  tablecell(t) {
    let r = this.parser.parseInline(t.tokens), i = t.header ? "th" : "td";
    return (t.align ? `<${i} align="${t.align}">` : `<${i}>`) + r + `</${i}>
`;
  }
  strong({ tokens: t }) {
    return `<strong>${this.parser.parseInline(t)}</strong>`;
  }
  em({ tokens: t }) {
    return `<em>${this.parser.parseInline(t)}</em>`;
  }
  codespan({ text: t }) {
    return `<code>${Zt(t, !0)}</code>`;
  }
  br(t) {
    return "<br>";
  }
  del({ tokens: t }) {
    return `<del>${this.parser.parseInline(t)}</del>`;
  }
  link({ href: t, title: r, tokens: i }) {
    let a = this.parser.parseInline(i), n = cl(t);
    if (n === null) return a;
    t = n;
    let o = '<a href="' + t + '"';
    return r && (o += ' title="' + Zt(r) + '"'), o += ">" + a + "</a>", o;
  }
  image({ href: t, title: r, text: i, tokens: a }) {
    a && (i = this.parser.parseInline(a, this.parser.textRenderer));
    let n = cl(t);
    if (n === null) return Zt(i);
    t = n;
    let o = `<img src="${t}" alt="${i}"`;
    return r && (o += ` title="${Zt(r)}"`), o += ">", o;
  }
  text(t) {
    return "tokens" in t && t.tokens ? this.parser.parseInline(t.tokens) : "escaped" in t && t.escaped ? t.text : Zt(t.text);
  }
}, ro = class {
  strong({ text: t }) {
    return t;
  }
  em({ text: t }) {
    return t;
  }
  codespan({ text: t }) {
    return t;
  }
  del({ text: t }) {
    return t;
  }
  html({ text: t }) {
    return t;
  }
  text({ text: t }) {
    return t;
  }
  link({ text: t }) {
    return "" + t;
  }
  image({ text: t }) {
    return "" + t;
  }
  br() {
    return "";
  }
}, he = class Xn {
  constructor(t) {
    ot(this, "options");
    ot(this, "renderer");
    ot(this, "textRenderer");
    this.options = t || Ke, this.options.renderer = this.options.renderer || new ma(), this.renderer = this.options.renderer, this.renderer.options = this.options, this.renderer.parser = this, this.textRenderer = new ro();
  }
  static parse(t, r) {
    return new Xn(r).parse(t);
  }
  static parseInline(t, r) {
    return new Xn(r).parseInline(t);
  }
  parse(t, r = !0) {
    var a, n;
    let i = "";
    for (let o = 0; o < t.length; o++) {
      let s = t[o];
      if ((n = (a = this.options.extensions) == null ? void 0 : a.renderers) != null && n[s.type]) {
        let c = s, h = this.options.extensions.renderers[c.type].call({ parser: this }, c);
        if (h !== !1 || !["space", "hr", "heading", "code", "table", "blockquote", "list", "html", "paragraph", "text"].includes(c.type)) {
          i += h || "";
          continue;
        }
      }
      let l = s;
      switch (l.type) {
        case "space": {
          i += this.renderer.space(l);
          continue;
        }
        case "hr": {
          i += this.renderer.hr(l);
          continue;
        }
        case "heading": {
          i += this.renderer.heading(l);
          continue;
        }
        case "code": {
          i += this.renderer.code(l);
          continue;
        }
        case "table": {
          i += this.renderer.table(l);
          continue;
        }
        case "blockquote": {
          i += this.renderer.blockquote(l);
          continue;
        }
        case "list": {
          i += this.renderer.list(l);
          continue;
        }
        case "html": {
          i += this.renderer.html(l);
          continue;
        }
        case "paragraph": {
          i += this.renderer.paragraph(l);
          continue;
        }
        case "text": {
          let c = l, h = this.renderer.text(c);
          for (; o + 1 < t.length && t[o + 1].type === "text"; ) c = t[++o], h += `
` + this.renderer.text(c);
          r ? i += this.renderer.paragraph({ type: "paragraph", raw: h, text: h, tokens: [{ type: "text", raw: h, text: h, escaped: !0 }] }) : i += h;
          continue;
        }
        default: {
          let c = 'Token with "' + l.type + '" type was not found.';
          if (this.options.silent) return console.error(c), "";
          throw new Error(c);
        }
      }
    }
    return i;
  }
  parseInline(t, r = this.renderer) {
    var a, n;
    let i = "";
    for (let o = 0; o < t.length; o++) {
      let s = t[o];
      if ((n = (a = this.options.extensions) == null ? void 0 : a.renderers) != null && n[s.type]) {
        let c = this.options.extensions.renderers[s.type].call({ parser: this }, s);
        if (c !== !1 || !["escape", "html", "link", "image", "strong", "em", "codespan", "br", "del", "text"].includes(s.type)) {
          i += c || "";
          continue;
        }
      }
      let l = s;
      switch (l.type) {
        case "escape": {
          i += r.text(l);
          break;
        }
        case "html": {
          i += r.html(l);
          break;
        }
        case "link": {
          i += r.link(l);
          break;
        }
        case "image": {
          i += r.image(l);
          break;
        }
        case "strong": {
          i += r.strong(l);
          break;
        }
        case "em": {
          i += r.em(l);
          break;
        }
        case "codespan": {
          i += r.codespan(l);
          break;
        }
        case "br": {
          i += r.br(l);
          break;
        }
        case "del": {
          i += r.del(l);
          break;
        }
        case "text": {
          i += r.text(l);
          break;
        }
        default: {
          let c = 'Token with "' + l.type + '" type was not found.';
          if (this.options.silent) return console.error(c), "";
          throw new Error(c);
        }
      }
    }
    return i;
  }
}, pn, Pi = (pn = class {
  constructor(t) {
    ot(this, "options");
    ot(this, "block");
    this.options = t || Ke;
  }
  preprocess(t) {
    return t;
  }
  postprocess(t) {
    return t;
  }
  processAllTokens(t) {
    return t;
  }
  provideLexer() {
    return this.block ? ce.lex : ce.lexInline;
  }
  provideParser() {
    return this.block ? he.parse : he.parseInline;
  }
}, ot(pn, "passThroughHooks", /* @__PURE__ */ new Set(["preprocess", "postprocess", "processAllTokens"])), pn), Nk = class {
  constructor(...t) {
    ot(this, "defaults", Xs());
    ot(this, "options", this.setOptions);
    ot(this, "parse", this.parseMarkdown(!0));
    ot(this, "parseInline", this.parseMarkdown(!1));
    ot(this, "Parser", he);
    ot(this, "Renderer", ma);
    ot(this, "TextRenderer", ro);
    ot(this, "Lexer", ce);
    ot(this, "Tokenizer", ga);
    ot(this, "Hooks", Pi);
    this.use(...t);
  }
  walkTokens(t, r) {
    var a, n;
    let i = [];
    for (let o of t) switch (i = i.concat(r.call(this, o)), o.type) {
      case "table": {
        let s = o;
        for (let l of s.header) i = i.concat(this.walkTokens(l.tokens, r));
        for (let l of s.rows) for (let c of l) i = i.concat(this.walkTokens(c.tokens, r));
        break;
      }
      case "list": {
        let s = o;
        i = i.concat(this.walkTokens(s.items, r));
        break;
      }
      default: {
        let s = o;
        (n = (a = this.defaults.extensions) == null ? void 0 : a.childTokens) != null && n[s.type] ? this.defaults.extensions.childTokens[s.type].forEach((l) => {
          let c = s[l].flat(1 / 0);
          i = i.concat(this.walkTokens(c, r));
        }) : s.tokens && (i = i.concat(this.walkTokens(s.tokens, r)));
      }
    }
    return i;
  }
  use(...t) {
    let r = this.defaults.extensions || { renderers: {}, childTokens: {} };
    return t.forEach((i) => {
      let a = { ...i };
      if (a.async = this.defaults.async || a.async || !1, i.extensions && (i.extensions.forEach((n) => {
        if (!n.name) throw new Error("extension name required");
        if ("renderer" in n) {
          let o = r.renderers[n.name];
          o ? r.renderers[n.name] = function(...s) {
            let l = n.renderer.apply(this, s);
            return l === !1 && (l = o.apply(this, s)), l;
          } : r.renderers[n.name] = n.renderer;
        }
        if ("tokenizer" in n) {
          if (!n.level || n.level !== "block" && n.level !== "inline") throw new Error("extension level must be 'block' or 'inline'");
          let o = r[n.level];
          o ? o.unshift(n.tokenizer) : r[n.level] = [n.tokenizer], n.start && (n.level === "block" ? r.startBlock ? r.startBlock.push(n.start) : r.startBlock = [n.start] : n.level === "inline" && (r.startInline ? r.startInline.push(n.start) : r.startInline = [n.start]));
        }
        "childTokens" in n && n.childTokens && (r.childTokens[n.name] = n.childTokens);
      }), a.extensions = r), i.renderer) {
        let n = this.defaults.renderer || new ma(this.defaults);
        for (let o in i.renderer) {
          if (!(o in n)) throw new Error(`renderer '${o}' does not exist`);
          if (["options", "parser"].includes(o)) continue;
          let s = o, l = i.renderer[s], c = n[s];
          n[s] = (...h) => {
            let u = l.apply(n, h);
            return u === !1 && (u = c.apply(n, h)), u || "";
          };
        }
        a.renderer = n;
      }
      if (i.tokenizer) {
        let n = this.defaults.tokenizer || new ga(this.defaults);
        for (let o in i.tokenizer) {
          if (!(o in n)) throw new Error(`tokenizer '${o}' does not exist`);
          if (["options", "rules", "lexer"].includes(o)) continue;
          let s = o, l = i.tokenizer[s], c = n[s];
          n[s] = (...h) => {
            let u = l.apply(n, h);
            return u === !1 && (u = c.apply(n, h)), u;
          };
        }
        a.tokenizer = n;
      }
      if (i.hooks) {
        let n = this.defaults.hooks || new Pi();
        for (let o in i.hooks) {
          if (!(o in n)) throw new Error(`hook '${o}' does not exist`);
          if (["options", "block"].includes(o)) continue;
          let s = o, l = i.hooks[s], c = n[s];
          Pi.passThroughHooks.has(o) ? n[s] = (h) => {
            if (this.defaults.async) return Promise.resolve(l.call(n, h)).then((f) => c.call(n, f));
            let u = l.call(n, h);
            return c.call(n, u);
          } : n[s] = (...h) => {
            let u = l.apply(n, h);
            return u === !1 && (u = c.apply(n, h)), u;
          };
        }
        a.hooks = n;
      }
      if (i.walkTokens) {
        let n = this.defaults.walkTokens, o = i.walkTokens;
        a.walkTokens = function(s) {
          let l = [];
          return l.push(o.call(this, s)), n && (l = l.concat(n.call(this, s))), l;
        };
      }
      this.defaults = { ...this.defaults, ...a };
    }), this;
  }
  setOptions(t) {
    return this.defaults = { ...this.defaults, ...t }, this;
  }
  lexer(t, r) {
    return ce.lex(t, r ?? this.defaults);
  }
  parser(t, r) {
    return he.parse(t, r ?? this.defaults);
  }
  parseMarkdown(t) {
    return (r, i) => {
      let a = { ...i }, n = { ...this.defaults, ...a }, o = this.onError(!!n.silent, !!n.async);
      if (this.defaults.async === !0 && a.async === !1) return o(new Error("marked(): The async option was set to true by an extension. Remove async: false from the parse options object to return a Promise."));
      if (typeof r > "u" || r === null) return o(new Error("marked(): input parameter is undefined or null"));
      if (typeof r != "string") return o(new Error("marked(): input parameter is of type " + Object.prototype.toString.call(r) + ", string expected"));
      n.hooks && (n.hooks.options = n, n.hooks.block = t);
      let s = n.hooks ? n.hooks.provideLexer() : t ? ce.lex : ce.lexInline, l = n.hooks ? n.hooks.provideParser() : t ? he.parse : he.parseInline;
      if (n.async) return Promise.resolve(n.hooks ? n.hooks.preprocess(r) : r).then((c) => s(c, n)).then((c) => n.hooks ? n.hooks.processAllTokens(c) : c).then((c) => n.walkTokens ? Promise.all(this.walkTokens(c, n.walkTokens)).then(() => c) : c).then((c) => l(c, n)).then((c) => n.hooks ? n.hooks.postprocess(c) : c).catch(o);
      try {
        n.hooks && (r = n.hooks.preprocess(r));
        let c = s(r, n);
        n.hooks && (c = n.hooks.processAllTokens(c)), n.walkTokens && this.walkTokens(c, n.walkTokens);
        let h = l(c, n);
        return n.hooks && (h = n.hooks.postprocess(h)), h;
      } catch (c) {
        return o(c);
      }
    };
  }
  onError(t, r) {
    return (i) => {
      if (i.message += `
Please report this to https://github.com/markedjs/marked.`, t) {
        let a = "<p>An error occurred:</p><pre>" + Zt(i.message + "", !0) + "</pre>";
        return r ? Promise.resolve(a) : a;
      }
      if (r) return Promise.reject(i);
      throw i;
    };
  }
}, Ye = new Nk();
function nt(e, t) {
  return Ye.parse(e, t);
}
nt.options = nt.setOptions = function(e) {
  return Ye.setOptions(e), nt.defaults = Ye.defaults, nf(nt.defaults), nt;
};
nt.getDefaults = Xs;
nt.defaults = Ke;
nt.use = function(...e) {
  return Ye.use(...e), nt.defaults = Ye.defaults, nf(nt.defaults), nt;
};
nt.walkTokens = function(e, t) {
  return Ye.walkTokens(e, t);
};
nt.parseInline = Ye.parseInline;
nt.Parser = he;
nt.parser = he.parse;
nt.Renderer = ma;
nt.TextRenderer = ro;
nt.Lexer = ce;
nt.lexer = ce.lex;
nt.Tokenizer = ga;
nt.Hooks = Pi;
nt.parse = nt;
nt.options;
nt.setOptions;
nt.use;
nt.walkTokens;
nt.parseInline;
he.parse;
ce.lex;
function mf(e) {
  for (var t = [], r = 1; r < arguments.length; r++)
    t[r - 1] = arguments[r];
  var i = Array.from(typeof e == "string" ? [e] : e);
  i[i.length - 1] = i[i.length - 1].replace(/\r?\n([\t ]*)$/, "");
  var a = i.reduce(function(s, l) {
    var c = l.match(/\n([\t ]+|(?!\s).)/g);
    return c ? s.concat(c.map(function(h) {
      var u, f;
      return (f = (u = h.match(/[\t ]/g)) === null || u === void 0 ? void 0 : u.length) !== null && f !== void 0 ? f : 0;
    })) : s;
  }, []);
  if (a.length) {
    var n = new RegExp(`
[	 ]{` + Math.min.apply(Math, a) + "}", "g");
    i = i.map(function(s) {
      return s.replace(n, `
`);
    });
  }
  i[0] = i[0].replace(/^\r?\n/, "");
  var o = i[0];
  return t.forEach(function(s, l) {
    var c = o.match(/(?:^|\n)( *)$/), h = c ? c[1] : "", u = s;
    typeof s == "string" && s.includes(`
`) && (u = String(s).split(`
`).map(function(f, d) {
      return d === 0 ? f : "" + h + f;
    }).join(`
`)), o += u + i[l + 1];
  }), o;
}
var zk = {
  body: '<g><rect width="80" height="80" style="fill: #087ebf; stroke-width: 0px;"/><text transform="translate(21.16 64.67)" style="fill: #fff; font-family: ArialMT, Arial; font-size: 67.75px;"><tspan x="0" y="0">?</tspan></text></g>',
  height: 80,
  width: 80
}, Vn = /* @__PURE__ */ new Map(), yf = /* @__PURE__ */ new Map(), qk = /* @__PURE__ */ p((e) => {
  for (const t of e) {
    if (!t.name)
      throw new Error(
        'Invalid icon loader. Must have a "name" property with non-empty string value.'
      );
    if (F.debug("Registering icon pack:", t.name), "loader" in t)
      yf.set(t.name, t.loader);
    else if ("icons" in t)
      Vn.set(t.name, t.icons);
    else
      throw F.error("Invalid icon loader:", t), new Error('Invalid icon loader. Must have either "icons" or "loader" property.');
  }
}, "registerIconPacks"), xf = /* @__PURE__ */ p(async (e, t) => {
  const r = WC(e, !0, t !== void 0);
  if (!r)
    throw new Error(`Invalid icon name: ${e}`);
  const i = r.prefix || t;
  if (!i)
    throw new Error(`Icon name must contain a prefix: ${e}`);
  let a = Vn.get(i);
  if (!a) {
    const o = yf.get(i);
    if (!o)
      throw new Error(`Icon set not found: ${r.prefix}`);
    try {
      a = { ...await o(), prefix: i }, Vn.set(i, a);
    } catch (s) {
      throw F.error(s), new Error(`Failed to load icon set: ${r.prefix}`);
    }
  }
  const n = YC(a, r.name);
  if (!n)
    throw new Error(`Icon not found: ${e}`);
  return n;
}, "getRegisteredIconData"), Wk = /* @__PURE__ */ p(async (e) => {
  try {
    return await xf(e), !0;
  } catch {
    return !1;
  }
}, "isIconAvailable"), yi = /* @__PURE__ */ p(async (e, t, r) => {
  let i;
  try {
    i = await xf(e, t == null ? void 0 : t.fallbackPrefix);
  } catch (o) {
    F.error(o), i = zk;
  }
  const a = QC(i, t);
  return ik(rk(a.body), {
    ...a.attributes,
    ...r
  });
}, "getIconSVG");
function bf(e, { markdownAutoWrap: t }) {
  const i = e.replace(/<br\/>/g, `
`).replace(/\n{2,}/g, `
`), a = mf(i);
  return t === !1 ? a.replace(/ /g, "&nbsp;") : a;
}
p(bf, "preprocessMarkdown");
function Cf(e, t = {}) {
  const r = bf(e, t), i = nt.lexer(r), a = [[]];
  let n = 0;
  function o(s, l = "normal") {
    s.type === "text" ? s.text.split(`
`).forEach((h, u) => {
      u !== 0 && (n++, a.push([])), h.split(" ").forEach((f) => {
        f = f.replace(/&#39;/g, "'"), f && a[n].push({ content: f, type: l });
      });
    }) : s.type === "strong" || s.type === "em" ? s.tokens.forEach((c) => {
      o(c, s.type);
    }) : s.type === "html" && a[n].push({ content: s.text, type: "normal" });
  }
  return p(o, "processNode"), i.forEach((s) => {
    var l;
    s.type === "paragraph" ? (l = s.tokens) == null || l.forEach((c) => {
      o(c);
    }) : s.type === "html" && a[n].push({ content: s.text, type: "normal" });
  }), a;
}
p(Cf, "markdownToLines");
function kf(e, { markdownAutoWrap: t } = {}) {
  const r = nt.lexer(e);
  function i(a) {
    var n, o, s;
    return a.type === "text" ? t === !1 ? a.text.replace(/\n */g, "<br/>").replace(/ /g, "&nbsp;") : a.text.replace(/\n */g, "<br/>") : a.type === "strong" ? `<strong>${(n = a.tokens) == null ? void 0 : n.map(i).join("")}</strong>` : a.type === "em" ? `<em>${(o = a.tokens) == null ? void 0 : o.map(i).join("")}</em>` : a.type === "paragraph" ? `<p>${(s = a.tokens) == null ? void 0 : s.map(i).join("")}</p>` : a.type === "space" ? "" : a.type === "html" ? `${a.text}` : a.type === "escape" ? a.text : `Unsupported markdown: ${a.type}`;
  }
  return p(i, "output"), r.map(i).join("");
}
p(kf, "markdownToHTML");
function wf(e) {
  return Intl.Segmenter ? [...new Intl.Segmenter().segment(e)].map((t) => t.segment) : [...e];
}
p(wf, "splitTextToChars");
function _f(e, t) {
  const r = wf(t.content);
  return io(e, [], r, t.type);
}
p(_f, "splitWordToFitWidth");
function io(e, t, r, i) {
  if (r.length === 0)
    return [
      { content: t.join(""), type: i },
      { content: "", type: i }
    ];
  const [a, ...n] = r, o = [...t, a];
  return e([{ content: o.join(""), type: i }]) ? io(e, o, n, i) : (t.length === 0 && a && (t.push(a), r.shift()), [
    { content: t.join(""), type: i },
    { content: r.join(""), type: i }
  ]);
}
p(io, "splitWordToFitWidthRecursion");
function vf(e, t) {
  if (e.some(({ content: r }) => r.includes(`
`)))
    throw new Error("splitLineToFitWidth does not support newlines in the line");
  return ya(e, t);
}
p(vf, "splitLineToFitWidth");
function ya(e, t, r = [], i = []) {
  if (e.length === 0)
    return i.length > 0 && r.push(i), r.length > 0 ? r : [];
  let a = "";
  e[0].content === " " && (a = " ", e.shift());
  const n = e.shift() ?? { content: " ", type: "normal" }, o = [...i];
  if (a !== "" && o.push({ content: a, type: "normal" }), o.push(n), t(o))
    return ya(e, t, r, o);
  if (i.length > 0)
    r.push(i), e.unshift(n);
  else if (n.content) {
    const [s, l] = _f(t, n);
    r.push([s]), l.content && e.unshift(l);
  }
  return ya(e, t, r);
}
p(ya, "splitLineToFitWidthRecursion");
function Zn(e, t) {
  t && e.attr("style", t);
}
p(Zn, "applyStyle");
async function Sf(e, t, r, i, a = !1) {
  const n = e.append("foreignObject");
  n.attr("width", `${10 * r}px`), n.attr("height", `${10 * r}px`);
  const o = n.append("xhtml:div");
  let s = t.label;
  t.label && yr(t.label) && (s = await fs(t.label.replace(vr.lineBreakRegex, `
`), at()));
  const l = t.isNode ? "nodeLabel" : "edgeLabel", c = o.append("span");
  c.html(s), Zn(c, t.labelStyle), c.attr("class", `${l} ${i}`), Zn(o, t.labelStyle), o.style("display", "table-cell"), o.style("white-space", "nowrap"), o.style("line-height", "1.5"), o.style("max-width", r + "px"), o.style("text-align", "center"), o.attr("xmlns", "http://www.w3.org/1999/xhtml"), a && o.attr("class", "labelBkg");
  let h = o.node().getBoundingClientRect();
  return h.width === r && (o.style("display", "table"), o.style("white-space", "break-spaces"), o.style("width", r + "px"), h = o.node().getBoundingClientRect()), n.node();
}
p(Sf, "addHtmlSpan");
function ja(e, t, r) {
  return e.append("tspan").attr("class", "text-outer-tspan").attr("x", 0).attr("y", t * r - 0.1 + "em").attr("dy", r + "em");
}
p(ja, "createTspan");
function Tf(e, t, r) {
  const i = e.append("text"), a = ja(i, 1, t);
  Ya(a, r);
  const n = a.node().getComputedTextLength();
  return i.remove(), n;
}
p(Tf, "computeWidthOfText");
function Hk(e, t, r) {
  var o;
  const i = e.append("text"), a = ja(i, 1, t);
  Ya(a, [{ content: r, type: "normal" }]);
  const n = (o = a.node()) == null ? void 0 : o.getBoundingClientRect();
  return n && i.remove(), n;
}
p(Hk, "computeDimensionOfText");
function Bf(e, t, r, i = !1) {
  const n = t.append("g"), o = n.insert("rect").attr("class", "background").attr("style", "stroke: none"), s = n.append("text").attr("y", "-10.1");
  let l = 0;
  for (const c of r) {
    const h = /* @__PURE__ */ p((f) => Tf(n, 1.1, f) <= e, "checkWidth"), u = h(c) ? [c] : vf(c, h);
    for (const f of u) {
      const d = ja(s, l, 1.1);
      Ya(d, f), l++;
    }
  }
  if (i) {
    const c = s.node().getBBox(), h = 2;
    return o.attr("x", c.x - h).attr("y", c.y - h).attr("width", c.width + 2 * h).attr("height", c.height + 2 * h), n.node();
  } else
    return s.node();
}
p(Bf, "createFormattedText");
function Ya(e, t) {
  e.text(""), t.forEach((r, i) => {
    const a = e.append("tspan").attr("font-style", r.type === "em" ? "italic" : "normal").attr("class", "text-inner-tspan").attr("font-weight", r.type === "strong" ? "bold" : "normal");
    i === 0 ? a.text(r.content) : a.text(" " + r.content);
  });
}
p(Ya, "updateTextContentAndStyles");
async function Lf(e) {
  const t = [];
  e.replace(/(fa[bklrs]?):fa-([\w-]+)/g, (i, a, n) => (t.push(
    (async () => {
      const o = `${a}:${n}`;
      return await Wk(o) ? await yi(o, void 0, { class: "label-icon" }) : `<i class='${La(i).replace(":", " ")}'></i>`;
    })()
  ), i));
  const r = await Promise.all(t);
  return e.replace(/(fa[bklrs]?):fa-([\w-]+)/g, () => r.shift() ?? "");
}
p(Lf, "replaceIconSubstring");
var Le = /* @__PURE__ */ p(async (e, t = "", {
  style: r = "",
  isTitle: i = !1,
  classes: a = "",
  useHtmlLabels: n = !0,
  isNode: o = !0,
  width: s = 200,
  addSvgBackground: l = !1
} = {}, c) => {
  if (F.debug(
    "XYZ createText",
    t,
    r,
    i,
    a,
    n,
    o,
    "addSvgBackground: ",
    l
  ), n) {
    const h = kf(t, c), u = await Lf(Ze(h)), f = t.replace(/\\\\/g, "\\"), d = {
      isNode: o,
      label: yr(t) ? f : u,
      labelStyle: r.replace("fill:", "color:")
    };
    return await Sf(e, d, s, a, l);
  } else {
    const h = t.replace(/<br\s*\/?>/g, "<br/>"), u = Cf(h.replace("<br>", "<br/>"), c), f = Bf(
      s,
      e,
      u,
      t ? l : !1
    );
    if (o) {
      /stroke:/.exec(r) && (r = r.replace("stroke:", "lineColor:"));
      const d = r.replace(/stroke:[^;]+;?/g, "").replace(/stroke-width:[^;]+;?/g, "").replace(/fill:[^;]+;?/g, "").replace(/color:/g, "fill:");
      et(f).attr("style", d);
    } else {
      const d = r.replace(/stroke:[^;]+;?/g, "").replace(/stroke-width:[^;]+;?/g, "").replace(/fill:[^;]+;?/g, "").replace(/background:/g, "fill:");
      et(f).select("rect").attr("style", d.replace(/background:/g, "fill:"));
      const g = r.replace(/stroke:[^;]+;?/g, "").replace(/stroke-width:[^;]+;?/g, "").replace(/fill:[^;]+;?/g, "").replace(/color:/g, "fill:");
      et(f).select("text").attr("style", g);
    }
    return f;
  }
}, "createText");
function sn(e, t, r) {
  if (e && e.length) {
    const [i, a] = t, n = Math.PI / 180 * r, o = Math.cos(n), s = Math.sin(n);
    for (const l of e) {
      const [c, h] = l;
      l[0] = (c - i) * o - (h - a) * s + i, l[1] = (c - i) * s + (h - a) * o + a;
    }
  }
}
function jk(e, t) {
  return e[0] === t[0] && e[1] === t[1];
}
function Yk(e, t, r, i = 1) {
  const a = r, n = Math.max(t, 0.1), o = e[0] && e[0][0] && typeof e[0][0] == "number" ? [e] : e, s = [0, 0];
  if (a) for (const c of o) sn(c, s, a);
  const l = function(c, h, u) {
    const f = [];
    for (const b of c) {
      const k = [...b];
      jk(k[0], k[k.length - 1]) || k.push([k[0][0], k[0][1]]), k.length > 2 && f.push(k);
    }
    const d = [];
    h = Math.max(h, 0.1);
    const g = [];
    for (const b of f) for (let k = 0; k < b.length - 1; k++) {
      const S = b[k], w = b[k + 1];
      if (S[1] !== w[1]) {
        const C = Math.min(S[1], w[1]);
        g.push({ ymin: C, ymax: Math.max(S[1], w[1]), x: C === S[1] ? S[0] : w[0], islope: (w[0] - S[0]) / (w[1] - S[1]) });
      }
    }
    if (g.sort((b, k) => b.ymin < k.ymin ? -1 : b.ymin > k.ymin ? 1 : b.x < k.x ? -1 : b.x > k.x ? 1 : b.ymax === k.ymax ? 0 : (b.ymax - k.ymax) / Math.abs(b.ymax - k.ymax)), !g.length) return d;
    let m = [], y = g[0].ymin, x = 0;
    for (; m.length || g.length; ) {
      if (g.length) {
        let b = -1;
        for (let k = 0; k < g.length && !(g[k].ymin > y); k++) b = k;
        g.splice(0, b + 1).forEach((k) => {
          m.push({ s: y, edge: k });
        });
      }
      if (m = m.filter((b) => !(b.edge.ymax <= y)), m.sort((b, k) => b.edge.x === k.edge.x ? 0 : (b.edge.x - k.edge.x) / Math.abs(b.edge.x - k.edge.x)), (u !== 1 || x % h == 0) && m.length > 1) for (let b = 0; b < m.length; b += 2) {
        const k = b + 1;
        if (k >= m.length) break;
        const S = m[b].edge, w = m[k].edge;
        d.push([[Math.round(S.x), y], [Math.round(w.x), y]]);
      }
      y += u, m.forEach((b) => {
        b.edge.x = b.edge.x + u * b.edge.islope;
      }), x++;
    }
    return d;
  }(o, n, i);
  if (a) {
    for (const c of o) sn(c, s, -a);
    (function(c, h, u) {
      const f = [];
      c.forEach((d) => f.push(...d)), sn(f, h, u);
    })(l, s, -a);
  }
  return l;
}
function xi(e, t) {
  var r;
  const i = t.hachureAngle + 90;
  let a = t.hachureGap;
  a < 0 && (a = 4 * t.strokeWidth), a = Math.round(Math.max(a, 0.1));
  let n = 1;
  return t.roughness >= 1 && (((r = t.randomizer) === null || r === void 0 ? void 0 : r.next()) || Math.random()) > 0.7 && (n = a), Yk(e, a, i, n || 1);
}
class ao {
  constructor(t) {
    this.helper = t;
  }
  fillPolygons(t, r) {
    return this._fillPolygons(t, r);
  }
  _fillPolygons(t, r) {
    const i = xi(t, r);
    return { type: "fillSketch", ops: this.renderLines(i, r) };
  }
  renderLines(t, r) {
    const i = [];
    for (const a of t) i.push(...this.helper.doubleLineOps(a[0][0], a[0][1], a[1][0], a[1][1], r));
    return i;
  }
}
function Ga(e) {
  const t = e[0], r = e[1];
  return Math.sqrt(Math.pow(t[0] - r[0], 2) + Math.pow(t[1] - r[1], 2));
}
class Gk extends ao {
  fillPolygons(t, r) {
    let i = r.hachureGap;
    i < 0 && (i = 4 * r.strokeWidth), i = Math.max(i, 0.1);
    const a = xi(t, Object.assign({}, r, { hachureGap: i })), n = Math.PI / 180 * r.hachureAngle, o = [], s = 0.5 * i * Math.cos(n), l = 0.5 * i * Math.sin(n);
    for (const [c, h] of a) Ga([c, h]) && o.push([[c[0] - s, c[1] + l], [...h]], [[c[0] + s, c[1] - l], [...h]]);
    return { type: "fillSketch", ops: this.renderLines(o, r) };
  }
}
class Uk extends ao {
  fillPolygons(t, r) {
    const i = this._fillPolygons(t, r), a = Object.assign({}, r, { hachureAngle: r.hachureAngle + 90 }), n = this._fillPolygons(t, a);
    return i.ops = i.ops.concat(n.ops), i;
  }
}
class Xk {
  constructor(t) {
    this.helper = t;
  }
  fillPolygons(t, r) {
    const i = xi(t, r = Object.assign({}, r, { hachureAngle: 0 }));
    return this.dotsOnLines(i, r);
  }
  dotsOnLines(t, r) {
    const i = [];
    let a = r.hachureGap;
    a < 0 && (a = 4 * r.strokeWidth), a = Math.max(a, 0.1);
    let n = r.fillWeight;
    n < 0 && (n = r.strokeWidth / 2);
    const o = a / 4;
    for (const s of t) {
      const l = Ga(s), c = l / a, h = Math.ceil(c) - 1, u = l - h * a, f = (s[0][0] + s[1][0]) / 2 - a / 4, d = Math.min(s[0][1], s[1][1]);
      for (let g = 0; g < h; g++) {
        const m = d + u + g * a, y = f - o + 2 * Math.random() * o, x = m - o + 2 * Math.random() * o, b = this.helper.ellipse(y, x, n, n, r);
        i.push(...b.ops);
      }
    }
    return { type: "fillSketch", ops: i };
  }
}
class Vk {
  constructor(t) {
    this.helper = t;
  }
  fillPolygons(t, r) {
    const i = xi(t, r);
    return { type: "fillSketch", ops: this.dashedLine(i, r) };
  }
  dashedLine(t, r) {
    const i = r.dashOffset < 0 ? r.hachureGap < 0 ? 4 * r.strokeWidth : r.hachureGap : r.dashOffset, a = r.dashGap < 0 ? r.hachureGap < 0 ? 4 * r.strokeWidth : r.hachureGap : r.dashGap, n = [];
    return t.forEach((o) => {
      const s = Ga(o), l = Math.floor(s / (i + a)), c = (s + a - l * (i + a)) / 2;
      let h = o[0], u = o[1];
      h[0] > u[0] && (h = o[1], u = o[0]);
      const f = Math.atan((u[1] - h[1]) / (u[0] - h[0]));
      for (let d = 0; d < l; d++) {
        const g = d * (i + a), m = g + i, y = [h[0] + g * Math.cos(f) + c * Math.cos(f), h[1] + g * Math.sin(f) + c * Math.sin(f)], x = [h[0] + m * Math.cos(f) + c * Math.cos(f), h[1] + m * Math.sin(f) + c * Math.sin(f)];
        n.push(...this.helper.doubleLineOps(y[0], y[1], x[0], x[1], r));
      }
    }), n;
  }
}
class Zk {
  constructor(t) {
    this.helper = t;
  }
  fillPolygons(t, r) {
    const i = r.hachureGap < 0 ? 4 * r.strokeWidth : r.hachureGap, a = r.zigzagOffset < 0 ? i : r.zigzagOffset, n = xi(t, r = Object.assign({}, r, { hachureGap: i + a }));
    return { type: "fillSketch", ops: this.zigzagLines(n, a, r) };
  }
  zigzagLines(t, r, i) {
    const a = [];
    return t.forEach((n) => {
      const o = Ga(n), s = Math.round(o / (2 * r));
      let l = n[0], c = n[1];
      l[0] > c[0] && (l = n[1], c = n[0]);
      const h = Math.atan((c[1] - l[1]) / (c[0] - l[0]));
      for (let u = 0; u < s; u++) {
        const f = 2 * u * r, d = 2 * (u + 1) * r, g = Math.sqrt(2 * Math.pow(r, 2)), m = [l[0] + f * Math.cos(h), l[1] + f * Math.sin(h)], y = [l[0] + d * Math.cos(h), l[1] + d * Math.sin(h)], x = [m[0] + g * Math.cos(h + Math.PI / 4), m[1] + g * Math.sin(h + Math.PI / 4)];
        a.push(...this.helper.doubleLineOps(m[0], m[1], x[0], x[1], i), ...this.helper.doubleLineOps(x[0], x[1], y[0], y[1], i));
      }
    }), a;
  }
}
const Ot = {};
class Kk {
  constructor(t) {
    this.seed = t;
  }
  next() {
    return this.seed ? (2 ** 31 - 1 & (this.seed = Math.imul(48271, this.seed))) / 2 ** 31 : Math.random();
  }
}
const Qk = 0, on = 1, fl = 2, Si = { A: 7, a: 7, C: 6, c: 6, H: 1, h: 1, L: 2, l: 2, M: 2, m: 2, Q: 4, q: 4, S: 4, s: 4, T: 2, t: 2, V: 1, v: 1, Z: 0, z: 0 };
function ln(e, t) {
  return e.type === t;
}
function no(e) {
  const t = [], r = function(o) {
    const s = new Array();
    for (; o !== ""; ) if (o.match(/^([ \t\r\n,]+)/)) o = o.substr(RegExp.$1.length);
    else if (o.match(/^([aAcChHlLmMqQsStTvVzZ])/)) s[s.length] = { type: Qk, text: RegExp.$1 }, o = o.substr(RegExp.$1.length);
    else {
      if (!o.match(/^(([-+]?[0-9]+(\.[0-9]*)?|[-+]?\.[0-9]+)([eE][-+]?[0-9]+)?)/)) return [];
      s[s.length] = { type: on, text: `${parseFloat(RegExp.$1)}` }, o = o.substr(RegExp.$1.length);
    }
    return s[s.length] = { type: fl, text: "" }, s;
  }(e);
  let i = "BOD", a = 0, n = r[a];
  for (; !ln(n, fl); ) {
    let o = 0;
    const s = [];
    if (i === "BOD") {
      if (n.text !== "M" && n.text !== "m") return no("M0,0" + e);
      a++, o = Si[n.text], i = n.text;
    } else ln(n, on) ? o = Si[i] : (a++, o = Si[n.text], i = n.text);
    if (!(a + o < r.length)) throw new Error("Path data ended short");
    for (let l = a; l < a + o; l++) {
      const c = r[l];
      if (!ln(c, on)) throw new Error("Param not a number: " + i + "," + c.text);
      s[s.length] = +c.text;
    }
    if (typeof Si[i] != "number") throw new Error("Bad segment: " + i);
    {
      const l = { key: i, data: s };
      t.push(l), a += o, n = r[a], i === "M" && (i = "L"), i === "m" && (i = "l");
    }
  }
  return t;
}
function Mf(e) {
  let t = 0, r = 0, i = 0, a = 0;
  const n = [];
  for (const { key: o, data: s } of e) switch (o) {
    case "M":
      n.push({ key: "M", data: [...s] }), [t, r] = s, [i, a] = s;
      break;
    case "m":
      t += s[0], r += s[1], n.push({ key: "M", data: [t, r] }), i = t, a = r;
      break;
    case "L":
      n.push({ key: "L", data: [...s] }), [t, r] = s;
      break;
    case "l":
      t += s[0], r += s[1], n.push({ key: "L", data: [t, r] });
      break;
    case "C":
      n.push({ key: "C", data: [...s] }), t = s[4], r = s[5];
      break;
    case "c": {
      const l = s.map((c, h) => h % 2 ? c + r : c + t);
      n.push({ key: "C", data: l }), t = l[4], r = l[5];
      break;
    }
    case "Q":
      n.push({ key: "Q", data: [...s] }), t = s[2], r = s[3];
      break;
    case "q": {
      const l = s.map((c, h) => h % 2 ? c + r : c + t);
      n.push({ key: "Q", data: l }), t = l[2], r = l[3];
      break;
    }
    case "A":
      n.push({ key: "A", data: [...s] }), t = s[5], r = s[6];
      break;
    case "a":
      t += s[5], r += s[6], n.push({ key: "A", data: [s[0], s[1], s[2], s[3], s[4], t, r] });
      break;
    case "H":
      n.push({ key: "H", data: [...s] }), t = s[0];
      break;
    case "h":
      t += s[0], n.push({ key: "H", data: [t] });
      break;
    case "V":
      n.push({ key: "V", data: [...s] }), r = s[0];
      break;
    case "v":
      r += s[0], n.push({ key: "V", data: [r] });
      break;
    case "S":
      n.push({ key: "S", data: [...s] }), t = s[2], r = s[3];
      break;
    case "s": {
      const l = s.map((c, h) => h % 2 ? c + r : c + t);
      n.push({ key: "S", data: l }), t = l[2], r = l[3];
      break;
    }
    case "T":
      n.push({ key: "T", data: [...s] }), t = s[0], r = s[1];
      break;
    case "t":
      t += s[0], r += s[1], n.push({ key: "T", data: [t, r] });
      break;
    case "Z":
    case "z":
      n.push({ key: "Z", data: [] }), t = i, r = a;
  }
  return n;
}
function $f(e) {
  const t = [];
  let r = "", i = 0, a = 0, n = 0, o = 0, s = 0, l = 0;
  for (const { key: c, data: h } of e) {
    switch (c) {
      case "M":
        t.push({ key: "M", data: [...h] }), [i, a] = h, [n, o] = h;
        break;
      case "C":
        t.push({ key: "C", data: [...h] }), i = h[4], a = h[5], s = h[2], l = h[3];
        break;
      case "L":
        t.push({ key: "L", data: [...h] }), [i, a] = h;
        break;
      case "H":
        i = h[0], t.push({ key: "L", data: [i, a] });
        break;
      case "V":
        a = h[0], t.push({ key: "L", data: [i, a] });
        break;
      case "S": {
        let u = 0, f = 0;
        r === "C" || r === "S" ? (u = i + (i - s), f = a + (a - l)) : (u = i, f = a), t.push({ key: "C", data: [u, f, ...h] }), s = h[0], l = h[1], i = h[2], a = h[3];
        break;
      }
      case "T": {
        const [u, f] = h;
        let d = 0, g = 0;
        r === "Q" || r === "T" ? (d = i + (i - s), g = a + (a - l)) : (d = i, g = a);
        const m = i + 2 * (d - i) / 3, y = a + 2 * (g - a) / 3, x = u + 2 * (d - u) / 3, b = f + 2 * (g - f) / 3;
        t.push({ key: "C", data: [m, y, x, b, u, f] }), s = d, l = g, i = u, a = f;
        break;
      }
      case "Q": {
        const [u, f, d, g] = h, m = i + 2 * (u - i) / 3, y = a + 2 * (f - a) / 3, x = d + 2 * (u - d) / 3, b = g + 2 * (f - g) / 3;
        t.push({ key: "C", data: [m, y, x, b, d, g] }), s = u, l = f, i = d, a = g;
        break;
      }
      case "A": {
        const u = Math.abs(h[0]), f = Math.abs(h[1]), d = h[2], g = h[3], m = h[4], y = h[5], x = h[6];
        u === 0 || f === 0 ? (t.push({ key: "C", data: [i, a, y, x, y, x] }), i = y, a = x) : (i !== y || a !== x) && (Af(i, a, y, x, u, f, d, g, m).forEach(function(b) {
          t.push({ key: "C", data: b });
        }), i = y, a = x);
        break;
      }
      case "Z":
        t.push({ key: "Z", data: [] }), i = n, a = o;
    }
    r = c;
  }
  return t;
}
function Dr(e, t, r) {
  return [e * Math.cos(r) - t * Math.sin(r), e * Math.sin(r) + t * Math.cos(r)];
}
function Af(e, t, r, i, a, n, o, s, l, c) {
  const h = (u = o, Math.PI * u / 180);
  var u;
  let f = [], d = 0, g = 0, m = 0, y = 0;
  if (c) [d, g, m, y] = c;
  else {
    [e, t] = Dr(e, t, -h), [r, i] = Dr(r, i, -h);
    const D = (e - r) / 2, B = (t - i) / 2;
    let M = D * D / (a * a) + B * B / (n * n);
    M > 1 && (M = Math.sqrt(M), a *= M, n *= M);
    const T = a * a, A = n * n, L = T * A - T * B * B - A * D * D, N = T * B * B + A * D * D, U = (s === l ? -1 : 1) * Math.sqrt(Math.abs(L / N));
    m = U * a * B / n + (e + r) / 2, y = U * -n * D / a + (t + i) / 2, d = Math.asin(parseFloat(((t - y) / n).toFixed(9))), g = Math.asin(parseFloat(((i - y) / n).toFixed(9))), e < m && (d = Math.PI - d), r < m && (g = Math.PI - g), d < 0 && (d = 2 * Math.PI + d), g < 0 && (g = 2 * Math.PI + g), l && d > g && (d -= 2 * Math.PI), !l && g > d && (g -= 2 * Math.PI);
  }
  let x = g - d;
  if (Math.abs(x) > 120 * Math.PI / 180) {
    const D = g, B = r, M = i;
    g = l && g > d ? d + 120 * Math.PI / 180 * 1 : d + 120 * Math.PI / 180 * -1, f = Af(r = m + a * Math.cos(g), i = y + n * Math.sin(g), B, M, a, n, o, 0, l, [g, D, m, y]);
  }
  x = g - d;
  const b = Math.cos(d), k = Math.sin(d), S = Math.cos(g), w = Math.sin(g), C = Math.tan(x / 4), _ = 4 / 3 * a * C, E = 4 / 3 * n * C, R = [e, t], O = [e + _ * k, t - E * b], $ = [r + _ * w, i - E * S], I = [r, i];
  if (O[0] = 2 * R[0] - O[0], O[1] = 2 * R[1] - O[1], c) return [O, $, I].concat(f);
  {
    f = [O, $, I].concat(f);
    const D = [];
    for (let B = 0; B < f.length; B += 3) {
      const M = Dr(f[B][0], f[B][1], h), T = Dr(f[B + 1][0], f[B + 1][1], h), A = Dr(f[B + 2][0], f[B + 2][1], h);
      D.push([M[0], M[1], T[0], T[1], A[0], A[1]]);
    }
    return D;
  }
}
const Jk = { randOffset: function(e, t) {
  return V(e, t);
}, randOffsetWithRange: function(e, t, r) {
  return xa(e, t, r);
}, ellipse: function(e, t, r, i, a) {
  const n = Ef(r, i, a);
  return Kn(e, t, a, n).opset;
}, doubleLineOps: function(e, t, r, i, a) {
  return Se(e, t, r, i, a, !0);
} };
function Ff(e, t, r, i, a) {
  return { type: "path", ops: Se(e, t, r, i, a) };
}
function Ii(e, t, r) {
  const i = (e || []).length;
  if (i > 2) {
    const a = [];
    for (let n = 0; n < i - 1; n++) a.push(...Se(e[n][0], e[n][1], e[n + 1][0], e[n + 1][1], r));
    return t && a.push(...Se(e[i - 1][0], e[i - 1][1], e[0][0], e[0][1], r)), { type: "path", ops: a };
  }
  return i === 2 ? Ff(e[0][0], e[0][1], e[1][0], e[1][1], r) : { type: "path", ops: [] };
}
function tw(e, t, r, i, a) {
  return function(n, o) {
    return Ii(n, !0, o);
  }([[e, t], [e + r, t], [e + r, t + i], [e, t + i]], a);
}
function dl(e, t) {
  if (e.length) {
    const r = typeof e[0][0] == "number" ? [e] : e, i = Ti(r[0], 1 * (1 + 0.2 * t.roughness), t), a = t.disableMultiStroke ? [] : Ti(r[0], 1.5 * (1 + 0.22 * t.roughness), ml(t));
    for (let n = 1; n < r.length; n++) {
      const o = r[n];
      if (o.length) {
        const s = Ti(o, 1 * (1 + 0.2 * t.roughness), t), l = t.disableMultiStroke ? [] : Ti(o, 1.5 * (1 + 0.22 * t.roughness), ml(t));
        for (const c of s) c.op !== "move" && i.push(c);
        for (const c of l) c.op !== "move" && a.push(c);
      }
    }
    return { type: "path", ops: i.concat(a) };
  }
  return { type: "path", ops: [] };
}
function Ef(e, t, r) {
  const i = Math.sqrt(2 * Math.PI * Math.sqrt((Math.pow(e / 2, 2) + Math.pow(t / 2, 2)) / 2)), a = Math.ceil(Math.max(r.curveStepCount, r.curveStepCount / Math.sqrt(200) * i)), n = 2 * Math.PI / a;
  let o = Math.abs(e / 2), s = Math.abs(t / 2);
  const l = 1 - r.curveFitting;
  return o += V(o * l, r), s += V(s * l, r), { increment: n, rx: o, ry: s };
}
function Kn(e, t, r, i) {
  const [a, n] = yl(i.increment, e, t, i.rx, i.ry, 1, i.increment * xa(0.1, xa(0.4, 1, r), r), r);
  let o = ba(a, null, r);
  if (!r.disableMultiStroke && r.roughness !== 0) {
    const [s] = yl(i.increment, e, t, i.rx, i.ry, 1.5, 0, r), l = ba(s, null, r);
    o = o.concat(l);
  }
  return { estimatedPoints: n, opset: { type: "path", ops: o } };
}
function pl(e, t, r, i, a, n, o, s, l) {
  const c = e, h = t;
  let u = Math.abs(r / 2), f = Math.abs(i / 2);
  u += V(0.01 * u, l), f += V(0.01 * f, l);
  let d = a, g = n;
  for (; d < 0; ) d += 2 * Math.PI, g += 2 * Math.PI;
  g - d > 2 * Math.PI && (d = 0, g = 2 * Math.PI);
  const m = 2 * Math.PI / l.curveStepCount, y = Math.min(m / 2, (g - d) / 2), x = xl(y, c, h, u, f, d, g, 1, l);
  if (!l.disableMultiStroke) {
    const b = xl(y, c, h, u, f, d, g, 1.5, l);
    x.push(...b);
  }
  return o && (s ? x.push(...Se(c, h, c + u * Math.cos(d), h + f * Math.sin(d), l), ...Se(c, h, c + u * Math.cos(g), h + f * Math.sin(g), l)) : x.push({ op: "lineTo", data: [c, h] }, { op: "lineTo", data: [c + u * Math.cos(d), h + f * Math.sin(d)] })), { type: "path", ops: x };
}
function gl(e, t) {
  const r = $f(Mf(no(e))), i = [];
  let a = [0, 0], n = [0, 0];
  for (const { key: o, data: s } of r) switch (o) {
    case "M":
      n = [s[0], s[1]], a = [s[0], s[1]];
      break;
    case "L":
      i.push(...Se(n[0], n[1], s[0], s[1], t)), n = [s[0], s[1]];
      break;
    case "C": {
      const [l, c, h, u, f, d] = s;
      i.push(...ew(l, c, h, u, f, d, n, t)), n = [f, d];
      break;
    }
    case "Z":
      i.push(...Se(n[0], n[1], a[0], a[1], t)), n = [a[0], a[1]];
  }
  return { type: "path", ops: i };
}
function cn(e, t) {
  const r = [];
  for (const i of e) if (i.length) {
    const a = t.maxRandomnessOffset || 0, n = i.length;
    if (n > 2) {
      r.push({ op: "move", data: [i[0][0] + V(a, t), i[0][1] + V(a, t)] });
      for (let o = 1; o < n; o++) r.push({ op: "lineTo", data: [i[o][0] + V(a, t), i[o][1] + V(a, t)] });
    }
  }
  return { type: "fillPath", ops: r };
}
function Je(e, t) {
  return function(r, i) {
    let a = r.fillStyle || "hachure";
    if (!Ot[a]) switch (a) {
      case "zigzag":
        Ot[a] || (Ot[a] = new Gk(i));
        break;
      case "cross-hatch":
        Ot[a] || (Ot[a] = new Uk(i));
        break;
      case "dots":
        Ot[a] || (Ot[a] = new Xk(i));
        break;
      case "dashed":
        Ot[a] || (Ot[a] = new Vk(i));
        break;
      case "zigzag-line":
        Ot[a] || (Ot[a] = new Zk(i));
        break;
      default:
        a = "hachure", Ot[a] || (Ot[a] = new ao(i));
    }
    return Ot[a];
  }(t, Jk).fillPolygons(e, t);
}
function ml(e) {
  const t = Object.assign({}, e);
  return t.randomizer = void 0, e.seed && (t.seed = e.seed + 1), t;
}
function Of(e) {
  return e.randomizer || (e.randomizer = new Kk(e.seed || 0)), e.randomizer.next();
}
function xa(e, t, r, i = 1) {
  return r.roughness * i * (Of(r) * (t - e) + e);
}
function V(e, t, r = 1) {
  return xa(-e, e, t, r);
}
function Se(e, t, r, i, a, n = !1) {
  const o = n ? a.disableMultiStrokeFill : a.disableMultiStroke, s = Qn(e, t, r, i, a, !0, !1);
  if (o) return s;
  const l = Qn(e, t, r, i, a, !0, !0);
  return s.concat(l);
}
function Qn(e, t, r, i, a, n, o) {
  const s = Math.pow(e - r, 2) + Math.pow(t - i, 2), l = Math.sqrt(s);
  let c = 1;
  c = l < 200 ? 1 : l > 500 ? 0.4 : -16668e-7 * l + 1.233334;
  let h = a.maxRandomnessOffset || 0;
  h * h * 100 > s && (h = l / 10);
  const u = h / 2, f = 0.2 + 0.2 * Of(a);
  let d = a.bowing * a.maxRandomnessOffset * (i - t) / 200, g = a.bowing * a.maxRandomnessOffset * (e - r) / 200;
  d = V(d, a, c), g = V(g, a, c);
  const m = [], y = () => V(u, a, c), x = () => V(h, a, c), b = a.preserveVertices;
  return o ? m.push({ op: "move", data: [e + (b ? 0 : y()), t + (b ? 0 : y())] }) : m.push({ op: "move", data: [e + (b ? 0 : V(h, a, c)), t + (b ? 0 : V(h, a, c))] }), o ? m.push({ op: "bcurveTo", data: [d + e + (r - e) * f + y(), g + t + (i - t) * f + y(), d + e + 2 * (r - e) * f + y(), g + t + 2 * (i - t) * f + y(), r + (b ? 0 : y()), i + (b ? 0 : y())] }) : m.push({ op: "bcurveTo", data: [d + e + (r - e) * f + x(), g + t + (i - t) * f + x(), d + e + 2 * (r - e) * f + x(), g + t + 2 * (i - t) * f + x(), r + (b ? 0 : x()), i + (b ? 0 : x())] }), m;
}
function Ti(e, t, r) {
  if (!e.length) return [];
  const i = [];
  i.push([e[0][0] + V(t, r), e[0][1] + V(t, r)]), i.push([e[0][0] + V(t, r), e[0][1] + V(t, r)]);
  for (let a = 1; a < e.length; a++) i.push([e[a][0] + V(t, r), e[a][1] + V(t, r)]), a === e.length - 1 && i.push([e[a][0] + V(t, r), e[a][1] + V(t, r)]);
  return ba(i, null, r);
}
function ba(e, t, r) {
  const i = e.length, a = [];
  if (i > 3) {
    const n = [], o = 1 - r.curveTightness;
    a.push({ op: "move", data: [e[1][0], e[1][1]] });
    for (let s = 1; s + 2 < i; s++) {
      const l = e[s];
      n[0] = [l[0], l[1]], n[1] = [l[0] + (o * e[s + 1][0] - o * e[s - 1][0]) / 6, l[1] + (o * e[s + 1][1] - o * e[s - 1][1]) / 6], n[2] = [e[s + 1][0] + (o * e[s][0] - o * e[s + 2][0]) / 6, e[s + 1][1] + (o * e[s][1] - o * e[s + 2][1]) / 6], n[3] = [e[s + 1][0], e[s + 1][1]], a.push({ op: "bcurveTo", data: [n[1][0], n[1][1], n[2][0], n[2][1], n[3][0], n[3][1]] });
    }
  } else i === 3 ? (a.push({ op: "move", data: [e[1][0], e[1][1]] }), a.push({ op: "bcurveTo", data: [e[1][0], e[1][1], e[2][0], e[2][1], e[2][0], e[2][1]] })) : i === 2 && a.push(...Qn(e[0][0], e[0][1], e[1][0], e[1][1], r, !0, !0));
  return a;
}
function yl(e, t, r, i, a, n, o, s) {
  const l = [], c = [];
  if (s.roughness === 0) {
    e /= 4, c.push([t + i * Math.cos(-e), r + a * Math.sin(-e)]);
    for (let h = 0; h <= 2 * Math.PI; h += e) {
      const u = [t + i * Math.cos(h), r + a * Math.sin(h)];
      l.push(u), c.push(u);
    }
    c.push([t + i * Math.cos(0), r + a * Math.sin(0)]), c.push([t + i * Math.cos(e), r + a * Math.sin(e)]);
  } else {
    const h = V(0.5, s) - Math.PI / 2;
    c.push([V(n, s) + t + 0.9 * i * Math.cos(h - e), V(n, s) + r + 0.9 * a * Math.sin(h - e)]);
    const u = 2 * Math.PI + h - 0.01;
    for (let f = h; f < u; f += e) {
      const d = [V(n, s) + t + i * Math.cos(f), V(n, s) + r + a * Math.sin(f)];
      l.push(d), c.push(d);
    }
    c.push([V(n, s) + t + i * Math.cos(h + 2 * Math.PI + 0.5 * o), V(n, s) + r + a * Math.sin(h + 2 * Math.PI + 0.5 * o)]), c.push([V(n, s) + t + 0.98 * i * Math.cos(h + o), V(n, s) + r + 0.98 * a * Math.sin(h + o)]), c.push([V(n, s) + t + 0.9 * i * Math.cos(h + 0.5 * o), V(n, s) + r + 0.9 * a * Math.sin(h + 0.5 * o)]);
  }
  return [c, l];
}
function xl(e, t, r, i, a, n, o, s, l) {
  const c = n + V(0.1, l), h = [];
  h.push([V(s, l) + t + 0.9 * i * Math.cos(c - e), V(s, l) + r + 0.9 * a * Math.sin(c - e)]);
  for (let u = c; u <= o; u += e) h.push([V(s, l) + t + i * Math.cos(u), V(s, l) + r + a * Math.sin(u)]);
  return h.push([t + i * Math.cos(o), r + a * Math.sin(o)]), h.push([t + i * Math.cos(o), r + a * Math.sin(o)]), ba(h, null, l);
}
function ew(e, t, r, i, a, n, o, s) {
  const l = [], c = [s.maxRandomnessOffset || 1, (s.maxRandomnessOffset || 1) + 0.3];
  let h = [0, 0];
  const u = s.disableMultiStroke ? 1 : 2, f = s.preserveVertices;
  for (let d = 0; d < u; d++) d === 0 ? l.push({ op: "move", data: [o[0], o[1]] }) : l.push({ op: "move", data: [o[0] + (f ? 0 : V(c[0], s)), o[1] + (f ? 0 : V(c[0], s))] }), h = f ? [a, n] : [a + V(c[d], s), n + V(c[d], s)], l.push({ op: "bcurveTo", data: [e + V(c[d], s), t + V(c[d], s), r + V(c[d], s), i + V(c[d], s), h[0], h[1]] });
  return l;
}
function Rr(e) {
  return [...e];
}
function bl(e, t = 0) {
  const r = e.length;
  if (r < 3) throw new Error("A curve must have at least three points.");
  const i = [];
  if (r === 3) i.push(Rr(e[0]), Rr(e[1]), Rr(e[2]), Rr(e[2]));
  else {
    const a = [];
    a.push(e[0], e[0]);
    for (let s = 1; s < e.length; s++) a.push(e[s]), s === e.length - 1 && a.push(e[s]);
    const n = [], o = 1 - t;
    i.push(Rr(a[0]));
    for (let s = 1; s + 2 < a.length; s++) {
      const l = a[s];
      n[0] = [l[0], l[1]], n[1] = [l[0] + (o * a[s + 1][0] - o * a[s - 1][0]) / 6, l[1] + (o * a[s + 1][1] - o * a[s - 1][1]) / 6], n[2] = [a[s + 1][0] + (o * a[s][0] - o * a[s + 2][0]) / 6, a[s + 1][1] + (o * a[s][1] - o * a[s + 2][1]) / 6], n[3] = [a[s + 1][0], a[s + 1][1]], i.push(n[1], n[2], n[3]);
    }
  }
  return i;
}
function Ni(e, t) {
  return Math.pow(e[0] - t[0], 2) + Math.pow(e[1] - t[1], 2);
}
function rw(e, t, r) {
  const i = Ni(t, r);
  if (i === 0) return Ni(e, t);
  let a = ((e[0] - t[0]) * (r[0] - t[0]) + (e[1] - t[1]) * (r[1] - t[1])) / i;
  return a = Math.max(0, Math.min(1, a)), Ni(e, Ae(t, r, a));
}
function Ae(e, t, r) {
  return [e[0] + (t[0] - e[0]) * r, e[1] + (t[1] - e[1]) * r];
}
function Jn(e, t, r, i) {
  const a = i || [];
  if (function(s, l) {
    const c = s[l + 0], h = s[l + 1], u = s[l + 2], f = s[l + 3];
    let d = 3 * h[0] - 2 * c[0] - f[0];
    d *= d;
    let g = 3 * h[1] - 2 * c[1] - f[1];
    g *= g;
    let m = 3 * u[0] - 2 * f[0] - c[0];
    m *= m;
    let y = 3 * u[1] - 2 * f[1] - c[1];
    return y *= y, d < m && (d = m), g < y && (g = y), d + g;
  }(e, t) < r) {
    const s = e[t + 0];
    a.length ? (n = a[a.length - 1], o = s, Math.sqrt(Ni(n, o)) > 1 && a.push(s)) : a.push(s), a.push(e[t + 3]);
  } else {
    const l = e[t + 0], c = e[t + 1], h = e[t + 2], u = e[t + 3], f = Ae(l, c, 0.5), d = Ae(c, h, 0.5), g = Ae(h, u, 0.5), m = Ae(f, d, 0.5), y = Ae(d, g, 0.5), x = Ae(m, y, 0.5);
    Jn([l, f, m, x], 0, r, a), Jn([x, y, g, u], 0, r, a);
  }
  var n, o;
  return a;
}
function iw(e, t) {
  return Ca(e, 0, e.length, t);
}
function Ca(e, t, r, i, a) {
  const n = a || [], o = e[t], s = e[r - 1];
  let l = 0, c = 1;
  for (let h = t + 1; h < r - 1; ++h) {
    const u = rw(e[h], o, s);
    u > l && (l = u, c = h);
  }
  return Math.sqrt(l) > i ? (Ca(e, t, c + 1, i, n), Ca(e, c, r, i, n)) : (n.length || n.push(o), n.push(s)), n;
}
function hn(e, t = 0.15, r) {
  const i = [], a = (e.length - 1) / 3;
  for (let n = 0; n < a; n++)
    Jn(e, 3 * n, t, i);
  return r && r > 0 ? Ca(i, 0, i.length, r) : i;
}
const Nt = "none";
class ka {
  constructor(t) {
    this.defaultOptions = { maxRandomnessOffset: 2, roughness: 1, bowing: 1, stroke: "#000", strokeWidth: 1, curveTightness: 0, curveFitting: 0.95, curveStepCount: 9, fillStyle: "hachure", fillWeight: -1, hachureAngle: -41, hachureGap: -1, dashOffset: -1, dashGap: -1, zigzagOffset: -1, seed: 0, disableMultiStroke: !1, disableMultiStrokeFill: !1, preserveVertices: !1, fillShapeRoughnessGain: 0.8 }, this.config = t || {}, this.config.options && (this.defaultOptions = this._o(this.config.options));
  }
  static newSeed() {
    return Math.floor(Math.random() * 2 ** 31);
  }
  _o(t) {
    return t ? Object.assign({}, this.defaultOptions, t) : this.defaultOptions;
  }
  _d(t, r, i) {
    return { shape: t, sets: r || [], options: i || this.defaultOptions };
  }
  line(t, r, i, a, n) {
    const o = this._o(n);
    return this._d("line", [Ff(t, r, i, a, o)], o);
  }
  rectangle(t, r, i, a, n) {
    const o = this._o(n), s = [], l = tw(t, r, i, a, o);
    if (o.fill) {
      const c = [[t, r], [t + i, r], [t + i, r + a], [t, r + a]];
      o.fillStyle === "solid" ? s.push(cn([c], o)) : s.push(Je([c], o));
    }
    return o.stroke !== Nt && s.push(l), this._d("rectangle", s, o);
  }
  ellipse(t, r, i, a, n) {
    const o = this._o(n), s = [], l = Ef(i, a, o), c = Kn(t, r, o, l);
    if (o.fill) if (o.fillStyle === "solid") {
      const h = Kn(t, r, o, l).opset;
      h.type = "fillPath", s.push(h);
    } else s.push(Je([c.estimatedPoints], o));
    return o.stroke !== Nt && s.push(c.opset), this._d("ellipse", s, o);
  }
  circle(t, r, i, a) {
    const n = this.ellipse(t, r, i, i, a);
    return n.shape = "circle", n;
  }
  linearPath(t, r) {
    const i = this._o(r);
    return this._d("linearPath", [Ii(t, !1, i)], i);
  }
  arc(t, r, i, a, n, o, s = !1, l) {
    const c = this._o(l), h = [], u = pl(t, r, i, a, n, o, s, !0, c);
    if (s && c.fill) if (c.fillStyle === "solid") {
      const f = Object.assign({}, c);
      f.disableMultiStroke = !0;
      const d = pl(t, r, i, a, n, o, !0, !1, f);
      d.type = "fillPath", h.push(d);
    } else h.push(function(f, d, g, m, y, x, b) {
      const k = f, S = d;
      let w = Math.abs(g / 2), C = Math.abs(m / 2);
      w += V(0.01 * w, b), C += V(0.01 * C, b);
      let _ = y, E = x;
      for (; _ < 0; ) _ += 2 * Math.PI, E += 2 * Math.PI;
      E - _ > 2 * Math.PI && (_ = 0, E = 2 * Math.PI);
      const R = (E - _) / b.curveStepCount, O = [];
      for (let $ = _; $ <= E; $ += R) O.push([k + w * Math.cos($), S + C * Math.sin($)]);
      return O.push([k + w * Math.cos(E), S + C * Math.sin(E)]), O.push([k, S]), Je([O], b);
    }(t, r, i, a, n, o, c));
    return c.stroke !== Nt && h.push(u), this._d("arc", h, c);
  }
  curve(t, r) {
    const i = this._o(r), a = [], n = dl(t, i);
    if (i.fill && i.fill !== Nt) if (i.fillStyle === "solid") {
      const o = dl(t, Object.assign(Object.assign({}, i), { disableMultiStroke: !0, roughness: i.roughness ? i.roughness + i.fillShapeRoughnessGain : 0 }));
      a.push({ type: "fillPath", ops: this._mergedShape(o.ops) });
    } else {
      const o = [], s = t;
      if (s.length) {
        const l = typeof s[0][0] == "number" ? [s] : s;
        for (const c of l) c.length < 3 ? o.push(...c) : c.length === 3 ? o.push(...hn(bl([c[0], c[0], c[1], c[2]]), 10, (1 + i.roughness) / 2)) : o.push(...hn(bl(c), 10, (1 + i.roughness) / 2));
      }
      o.length && a.push(Je([o], i));
    }
    return i.stroke !== Nt && a.push(n), this._d("curve", a, i);
  }
  polygon(t, r) {
    const i = this._o(r), a = [], n = Ii(t, !0, i);
    return i.fill && (i.fillStyle === "solid" ? a.push(cn([t], i)) : a.push(Je([t], i))), i.stroke !== Nt && a.push(n), this._d("polygon", a, i);
  }
  path(t, r) {
    const i = this._o(r), a = [];
    if (!t) return this._d("path", a, i);
    t = (t || "").replace(/\n/g, " ").replace(/(-\s)/g, "-").replace("/(ss)/g", " ");
    const n = i.fill && i.fill !== "transparent" && i.fill !== Nt, o = i.stroke !== Nt, s = !!(i.simplification && i.simplification < 1), l = function(h, u, f) {
      const d = $f(Mf(no(h))), g = [];
      let m = [], y = [0, 0], x = [];
      const b = () => {
        x.length >= 4 && m.push(...hn(x, u)), x = [];
      }, k = () => {
        b(), m.length && (g.push(m), m = []);
      };
      for (const { key: w, data: C } of d) switch (w) {
        case "M":
          k(), y = [C[0], C[1]], m.push(y);
          break;
        case "L":
          b(), m.push([C[0], C[1]]);
          break;
        case "C":
          if (!x.length) {
            const _ = m.length ? m[m.length - 1] : y;
            x.push([_[0], _[1]]);
          }
          x.push([C[0], C[1]]), x.push([C[2], C[3]]), x.push([C[4], C[5]]);
          break;
        case "Z":
          b(), m.push([y[0], y[1]]);
      }
      if (k(), !f) return g;
      const S = [];
      for (const w of g) {
        const C = iw(w, f);
        C.length && S.push(C);
      }
      return S;
    }(t, 1, s ? 4 - 4 * (i.simplification || 1) : (1 + i.roughness) / 2), c = gl(t, i);
    if (n) if (i.fillStyle === "solid") if (l.length === 1) {
      const h = gl(t, Object.assign(Object.assign({}, i), { disableMultiStroke: !0, roughness: i.roughness ? i.roughness + i.fillShapeRoughnessGain : 0 }));
      a.push({ type: "fillPath", ops: this._mergedShape(h.ops) });
    } else a.push(cn(l, i));
    else a.push(Je(l, i));
    return o && (s ? l.forEach((h) => {
      a.push(Ii(h, !1, i));
    }) : a.push(c)), this._d("path", a, i);
  }
  opsToPath(t, r) {
    let i = "";
    for (const a of t.ops) {
      const n = typeof r == "number" && r >= 0 ? a.data.map((o) => +o.toFixed(r)) : a.data;
      switch (a.op) {
        case "move":
          i += `M${n[0]} ${n[1]} `;
          break;
        case "bcurveTo":
          i += `C${n[0]} ${n[1]}, ${n[2]} ${n[3]}, ${n[4]} ${n[5]} `;
          break;
        case "lineTo":
          i += `L${n[0]} ${n[1]} `;
      }
    }
    return i.trim();
  }
  toPaths(t) {
    const r = t.sets || [], i = t.options || this.defaultOptions, a = [];
    for (const n of r) {
      let o = null;
      switch (n.type) {
        case "path":
          o = { d: this.opsToPath(n), stroke: i.stroke, strokeWidth: i.strokeWidth, fill: Nt };
          break;
        case "fillPath":
          o = { d: this.opsToPath(n), stroke: Nt, strokeWidth: 0, fill: i.fill || Nt };
          break;
        case "fillSketch":
          o = this.fillSketch(n, i);
      }
      o && a.push(o);
    }
    return a;
  }
  fillSketch(t, r) {
    let i = r.fillWeight;
    return i < 0 && (i = r.strokeWidth / 2), { d: this.opsToPath(t), stroke: r.fill || Nt, strokeWidth: i, fill: Nt };
  }
  _mergedShape(t) {
    return t.filter((r, i) => i === 0 || r.op !== "move");
  }
}
class aw {
  constructor(t, r) {
    this.canvas = t, this.ctx = this.canvas.getContext("2d"), this.gen = new ka(r);
  }
  draw(t) {
    const r = t.sets || [], i = t.options || this.getDefaultOptions(), a = this.ctx, n = t.options.fixedDecimalPlaceDigits;
    for (const o of r) switch (o.type) {
      case "path":
        a.save(), a.strokeStyle = i.stroke === "none" ? "transparent" : i.stroke, a.lineWidth = i.strokeWidth, i.strokeLineDash && a.setLineDash(i.strokeLineDash), i.strokeLineDashOffset && (a.lineDashOffset = i.strokeLineDashOffset), this._drawToContext(a, o, n), a.restore();
        break;
      case "fillPath": {
        a.save(), a.fillStyle = i.fill || "";
        const s = t.shape === "curve" || t.shape === "polygon" || t.shape === "path" ? "evenodd" : "nonzero";
        this._drawToContext(a, o, n, s), a.restore();
        break;
      }
      case "fillSketch":
        this.fillSketch(a, o, i);
    }
  }
  fillSketch(t, r, i) {
    let a = i.fillWeight;
    a < 0 && (a = i.strokeWidth / 2), t.save(), i.fillLineDash && t.setLineDash(i.fillLineDash), i.fillLineDashOffset && (t.lineDashOffset = i.fillLineDashOffset), t.strokeStyle = i.fill || "", t.lineWidth = a, this._drawToContext(t, r, i.fixedDecimalPlaceDigits), t.restore();
  }
  _drawToContext(t, r, i, a = "nonzero") {
    t.beginPath();
    for (const n of r.ops) {
      const o = typeof i == "number" && i >= 0 ? n.data.map((s) => +s.toFixed(i)) : n.data;
      switch (n.op) {
        case "move":
          t.moveTo(o[0], o[1]);
          break;
        case "bcurveTo":
          t.bezierCurveTo(o[0], o[1], o[2], o[3], o[4], o[5]);
          break;
        case "lineTo":
          t.lineTo(o[0], o[1]);
      }
    }
    r.type === "fillPath" ? t.fill(a) : t.stroke();
  }
  get generator() {
    return this.gen;
  }
  getDefaultOptions() {
    return this.gen.defaultOptions;
  }
  line(t, r, i, a, n) {
    const o = this.gen.line(t, r, i, a, n);
    return this.draw(o), o;
  }
  rectangle(t, r, i, a, n) {
    const o = this.gen.rectangle(t, r, i, a, n);
    return this.draw(o), o;
  }
  ellipse(t, r, i, a, n) {
    const o = this.gen.ellipse(t, r, i, a, n);
    return this.draw(o), o;
  }
  circle(t, r, i, a) {
    const n = this.gen.circle(t, r, i, a);
    return this.draw(n), n;
  }
  linearPath(t, r) {
    const i = this.gen.linearPath(t, r);
    return this.draw(i), i;
  }
  polygon(t, r) {
    const i = this.gen.polygon(t, r);
    return this.draw(i), i;
  }
  arc(t, r, i, a, n, o, s = !1, l) {
    const c = this.gen.arc(t, r, i, a, n, o, s, l);
    return this.draw(c), c;
  }
  curve(t, r) {
    const i = this.gen.curve(t, r);
    return this.draw(i), i;
  }
  path(t, r) {
    const i = this.gen.path(t, r);
    return this.draw(i), i;
  }
}
const Bi = "http://www.w3.org/2000/svg";
class nw {
  constructor(t, r) {
    this.svg = t, this.gen = new ka(r);
  }
  draw(t) {
    const r = t.sets || [], i = t.options || this.getDefaultOptions(), a = this.svg.ownerDocument || window.document, n = a.createElementNS(Bi, "g"), o = t.options.fixedDecimalPlaceDigits;
    for (const s of r) {
      let l = null;
      switch (s.type) {
        case "path":
          l = a.createElementNS(Bi, "path"), l.setAttribute("d", this.opsToPath(s, o)), l.setAttribute("stroke", i.stroke), l.setAttribute("stroke-width", i.strokeWidth + ""), l.setAttribute("fill", "none"), i.strokeLineDash && l.setAttribute("stroke-dasharray", i.strokeLineDash.join(" ").trim()), i.strokeLineDashOffset && l.setAttribute("stroke-dashoffset", `${i.strokeLineDashOffset}`);
          break;
        case "fillPath":
          l = a.createElementNS(Bi, "path"), l.setAttribute("d", this.opsToPath(s, o)), l.setAttribute("stroke", "none"), l.setAttribute("stroke-width", "0"), l.setAttribute("fill", i.fill || ""), t.shape !== "curve" && t.shape !== "polygon" || l.setAttribute("fill-rule", "evenodd");
          break;
        case "fillSketch":
          l = this.fillSketch(a, s, i);
      }
      l && n.appendChild(l);
    }
    return n;
  }
  fillSketch(t, r, i) {
    let a = i.fillWeight;
    a < 0 && (a = i.strokeWidth / 2);
    const n = t.createElementNS(Bi, "path");
    return n.setAttribute("d", this.opsToPath(r, i.fixedDecimalPlaceDigits)), n.setAttribute("stroke", i.fill || ""), n.setAttribute("stroke-width", a + ""), n.setAttribute("fill", "none"), i.fillLineDash && n.setAttribute("stroke-dasharray", i.fillLineDash.join(" ").trim()), i.fillLineDashOffset && n.setAttribute("stroke-dashoffset", `${i.fillLineDashOffset}`), n;
  }
  get generator() {
    return this.gen;
  }
  getDefaultOptions() {
    return this.gen.defaultOptions;
  }
  opsToPath(t, r) {
    return this.gen.opsToPath(t, r);
  }
  line(t, r, i, a, n) {
    const o = this.gen.line(t, r, i, a, n);
    return this.draw(o);
  }
  rectangle(t, r, i, a, n) {
    const o = this.gen.rectangle(t, r, i, a, n);
    return this.draw(o);
  }
  ellipse(t, r, i, a, n) {
    const o = this.gen.ellipse(t, r, i, a, n);
    return this.draw(o);
  }
  circle(t, r, i, a) {
    const n = this.gen.circle(t, r, i, a);
    return this.draw(n);
  }
  linearPath(t, r) {
    const i = this.gen.linearPath(t, r);
    return this.draw(i);
  }
  polygon(t, r) {
    const i = this.gen.polygon(t, r);
    return this.draw(i);
  }
  arc(t, r, i, a, n, o, s = !1, l) {
    const c = this.gen.arc(t, r, i, a, n, o, s, l);
    return this.draw(c);
  }
  curve(t, r) {
    const i = this.gen.curve(t, r);
    return this.draw(i);
  }
  path(t, r) {
    const i = this.gen.path(t, r);
    return this.draw(i);
  }
}
var W = { canvas: (e, t) => new aw(e, t), svg: (e, t) => new nw(e, t), generator: (e) => new ka(e), newSeed: () => ka.newSeed() }, Q = /* @__PURE__ */ p(async (e, t, r) => {
  var u, f;
  let i;
  const a = t.useHtmlLabels || bt((u = at()) == null ? void 0 : u.htmlLabels);
  r ? i = r : i = "node default";
  const n = e.insert("g").attr("class", i).attr("id", t.domId || t.id), o = n.insert("g").attr("class", "label").attr("style", Et(t.labelStyle));
  let s;
  t.label === void 0 ? s = "" : s = typeof t.label == "string" ? t.label : t.label[0];
  const l = await Le(o, qe(Ze(s), at()), {
    useHtmlLabels: a,
    width: t.width || ((f = at().flowchart) == null ? void 0 : f.wrappingWidth),
    // @ts-expect-error -- This is currently not used. Should this be `classes` instead?
    cssClasses: "markdown-node-label",
    style: t.labelStyle,
    addSvgBackground: !!t.icon || !!t.img
  });
  let c = l.getBBox();
  const h = ((t == null ? void 0 : t.padding) ?? 0) / 2;
  if (a) {
    const d = l.children[0], g = et(l), m = d.getElementsByTagName("img");
    if (m) {
      const y = s.replace(/<img[^>]*>/g, "").trim() === "";
      await Promise.all(
        [...m].map(
          (x) => new Promise((b) => {
            function k() {
              if (x.style.display = "flex", x.style.flexDirection = "column", y) {
                const S = at().fontSize ? at().fontSize : window.getComputedStyle(document.body).fontSize, w = 5, [C = Yl.fontSize] = qa(S), _ = C * w + "px";
                x.style.minWidth = _, x.style.maxWidth = _;
              } else
                x.style.width = "100%";
              b(x);
            }
            p(k, "setupImage"), setTimeout(() => {
              x.complete && k();
            }), x.addEventListener("error", k), x.addEventListener("load", k);
          })
        )
      );
    }
    c = d.getBoundingClientRect(), g.attr("width", c.width), g.attr("height", c.height);
  }
  return a ? o.attr("transform", "translate(" + -c.width / 2 + ", " + -c.height / 2 + ")") : o.attr("transform", "translate(0, " + -c.height / 2 + ")"), t.centerLabel && o.attr("transform", "translate(" + -c.width / 2 + ", " + -c.height / 2 + ")"), o.insert("rect", ":first-child"), { shapeSvg: n, bbox: c, halfPadding: h, label: o };
}, "labelHelper"), un = /* @__PURE__ */ p(async (e, t, r) => {
  var l, c, h, u, f, d;
  const i = r.useHtmlLabels || bt((c = (l = at()) == null ? void 0 : l.flowchart) == null ? void 0 : c.htmlLabels), a = e.insert("g").attr("class", "label").attr("style", r.labelStyle || ""), n = await Le(a, qe(Ze(t), at()), {
    useHtmlLabels: i,
    width: r.width || ((u = (h = at()) == null ? void 0 : h.flowchart) == null ? void 0 : u.wrappingWidth),
    style: r.labelStyle,
    addSvgBackground: !!r.icon || !!r.img
  });
  let o = n.getBBox();
  const s = r.padding / 2;
  if (bt((d = (f = at()) == null ? void 0 : f.flowchart) == null ? void 0 : d.htmlLabels)) {
    const g = n.children[0], m = et(n);
    o = g.getBoundingClientRect(), m.attr("width", o.width), m.attr("height", o.height);
  }
  return i ? a.attr("transform", "translate(" + -o.width / 2 + ", " + -o.height / 2 + ")") : a.attr("transform", "translate(0, " + -o.height / 2 + ")"), r.centerLabel && a.attr("transform", "translate(" + -o.width / 2 + ", " + -o.height / 2 + ")"), a.insert("rect", ":first-child"), { shapeSvg: e, bbox: o, halfPadding: s, label: a };
}, "insertLabel"), j = /* @__PURE__ */ p((e, t) => {
  const r = t.node().getBBox();
  e.width = r.width, e.height = r.height;
}, "updateNodeBounds"), Z = /* @__PURE__ */ p((e, t) => (e.look === "handDrawn" ? "rough-node" : "node") + " " + e.cssClasses + " " + (t || ""), "getNodeClasses");
function rt(e) {
  const t = e.map((r, i) => `${i === 0 ? "M" : "L"}${r.x},${r.y}`);
  return t.push("Z"), t.join(" ");
}
p(rt, "createPathFromPoints");
function Te(e, t, r, i, a, n) {
  const o = [], l = r - e, c = i - t, h = l / n, u = 2 * Math.PI / h, f = t + c / 2;
  for (let d = 0; d <= 50; d++) {
    const g = d / 50, m = e + g * l, y = f + a * Math.sin(u * (m - e));
    o.push({ x: m, y });
  }
  return o;
}
p(Te, "generateFullSineWavePoints");
function so(e, t, r, i, a, n) {
  const o = [], s = a * Math.PI / 180, h = (n * Math.PI / 180 - s) / (i - 1);
  for (let u = 0; u < i; u++) {
    const f = s + u * h, d = e + r * Math.cos(f), g = t + r * Math.sin(f);
    o.push({ x: -d, y: -g });
  }
  return o;
}
p(so, "generateCirclePoints");
var sw = /* @__PURE__ */ p((e, t) => {
  var r = e.x, i = e.y, a = t.x - r, n = t.y - i, o = e.width / 2, s = e.height / 2, l, c;
  return Math.abs(n) * o > Math.abs(a) * s ? (n < 0 && (s = -s), l = n === 0 ? 0 : s * a / n, c = s) : (a < 0 && (o = -o), l = o, c = a === 0 ? 0 : o * n / a), { x: r + l, y: i + c };
}, "intersectRect"), Lr = sw;
function Df(e, t) {
  t && e.attr("style", t);
}
p(Df, "applyStyle");
async function Rf(e) {
  const t = et(document.createElementNS("http://www.w3.org/2000/svg", "foreignObject")), r = t.append("xhtml:div");
  let i = e.label;
  e.label && yr(e.label) && (i = await fs(e.label.replace(vr.lineBreakRegex, `
`), at()));
  const a = e.isNode ? "nodeLabel" : "edgeLabel";
  return r.html(
    '<span class="' + a + '" ' + (e.labelStyle ? 'style="' + e.labelStyle + '"' : "") + // codeql [js/html-constructed-from-input] : false positive
    ">" + i + "</span>"
  ), Df(r, e.labelStyle), r.style("display", "inline-block"), r.style("padding-right", "1px"), r.style("white-space", "nowrap"), r.attr("xmlns", "http://www.w3.org/1999/xhtml"), t.node();
}
p(Rf, "addHtmlLabel");
var ow = /* @__PURE__ */ p(async (e, t, r, i) => {
  let a = e || "";
  if (typeof a == "object" && (a = a[0]), bt(at().flowchart.htmlLabels)) {
    a = a.replace(/\\n|\n/g, "<br />"), F.info("vertexText" + a);
    const n = {
      isNode: i,
      label: Ze(a).replace(
        /fa[blrs]?:fa-[\w-]+/g,
        (s) => `<i class='${s.replace(":", " ")}'></i>`
      ),
      labelStyle: t && t.replace("fill:", "color:")
    };
    return await Rf(n);
  } else {
    const n = document.createElementNS("http://www.w3.org/2000/svg", "text");
    n.setAttribute("style", t.replace("color:", "fill:"));
    let o = [];
    typeof a == "string" ? o = a.split(/\\n|\n|<br\s*\/?>/gi) : Array.isArray(a) ? o = a : o = [];
    for (const s of o) {
      const l = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
      l.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve"), l.setAttribute("dy", "1em"), l.setAttribute("x", "0"), r ? l.setAttribute("class", "title-row") : l.setAttribute("class", "row"), l.textContent = s.trim(), n.appendChild(l);
    }
    return n;
  }
}, "createLabel"), Pe = ow, me = /* @__PURE__ */ p((e, t, r, i, a) => [
  "M",
  e + a,
  t,
  // Move to the first point
  "H",
  e + r - a,
  // Draw horizontal line to the beginning of the right corner
  "A",
  a,
  a,
  0,
  0,
  1,
  e + r,
  t + a,
  // Draw arc to the right top corner
  "V",
  t + i - a,
  // Draw vertical line down to the beginning of the right bottom corner
  "A",
  a,
  a,
  0,
  0,
  1,
  e + r - a,
  t + i,
  // Draw arc to the right bottom corner
  "H",
  e + a,
  // Draw horizontal line to the beginning of the left bottom corner
  "A",
  a,
  a,
  0,
  0,
  1,
  e,
  t + i - a,
  // Draw arc to the left bottom corner
  "V",
  t + a,
  // Draw vertical line up to the beginning of the left top corner
  "A",
  a,
  a,
  0,
  0,
  1,
  e + a,
  t,
  // Draw arc to the left top corner
  "Z"
  // Close the path
].join(" "), "createRoundedRectPathD"), Pf = /* @__PURE__ */ p(async (e, t) => {
  F.info("Creating subgraph rect for ", t.id, t);
  const r = at(), { themeVariables: i, handDrawnSeed: a } = r, { clusterBkg: n, clusterBorder: o } = i, { labelStyles: s, nodeStyles: l, borderStyles: c, backgroundStyles: h } = Y(t), u = e.insert("g").attr("class", "cluster " + t.cssClasses).attr("id", t.id).attr("data-look", t.look), f = bt(r.flowchart.htmlLabels), d = u.insert("g").attr("class", "cluster-label "), g = await Le(d, t.label, {
    style: t.labelStyle,
    useHtmlLabels: f,
    isNode: !0
  });
  let m = g.getBBox();
  if (bt(r.flowchart.htmlLabels)) {
    const _ = g.children[0], E = et(g);
    m = _.getBoundingClientRect(), E.attr("width", m.width), E.attr("height", m.height);
  }
  const y = t.width <= m.width + t.padding ? m.width + t.padding : t.width;
  t.width <= m.width + t.padding ? t.diff = (y - t.width) / 2 - t.padding : t.diff = -t.padding;
  const x = t.height, b = t.x - y / 2, k = t.y - x / 2;
  F.trace("Data ", t, JSON.stringify(t));
  let S;
  if (t.look === "handDrawn") {
    const _ = W.svg(u), E = H(t, {
      roughness: 0.7,
      fill: n,
      // fill: 'red',
      stroke: o,
      fillWeight: 3,
      seed: a
    }), R = _.path(me(b, k, y, x, 0), E);
    S = u.insert(() => (F.debug("Rough node insert CXC", R), R), ":first-child"), S.select("path:nth-child(2)").attr("style", c.join(";")), S.select("path").attr("style", h.join(";").replace("fill", "stroke"));
  } else
    S = u.insert("rect", ":first-child"), S.attr("style", l).attr("rx", t.rx).attr("ry", t.ry).attr("x", b).attr("y", k).attr("width", y).attr("height", x);
  const { subGraphTitleTopMargin: w } = vs(r);
  if (d.attr(
    "transform",
    // This puts the label on top of the box instead of inside it
    `translate(${t.x - m.width / 2}, ${t.y - t.height / 2 + w})`
  ), s) {
    const _ = d.select("span");
    _ && _.attr("style", s);
  }
  const C = S.node().getBBox();
  return t.offsetX = 0, t.width = C.width, t.height = C.height, t.offsetY = m.height - t.padding / 2, t.intersect = function(_) {
    return Lr(t, _);
  }, { cluster: u, labelBBox: m };
}, "rect"), lw = /* @__PURE__ */ p((e, t) => {
  const r = e.insert("g").attr("class", "note-cluster").attr("id", t.id), i = r.insert("rect", ":first-child"), a = 0 * t.padding, n = a / 2;
  i.attr("rx", t.rx).attr("ry", t.ry).attr("x", t.x - t.width / 2 - n).attr("y", t.y - t.height / 2 - n).attr("width", t.width + a).attr("height", t.height + a).attr("fill", "none");
  const o = i.node().getBBox();
  return t.width = o.width, t.height = o.height, t.intersect = function(s) {
    return Lr(t, s);
  }, { cluster: r, labelBBox: { width: 0, height: 0 } };
}, "noteGroup"), cw = /* @__PURE__ */ p(async (e, t) => {
  const r = at(), { themeVariables: i, handDrawnSeed: a } = r, { altBackground: n, compositeBackground: o, compositeTitleBackground: s, nodeBorder: l } = i, c = e.insert("g").attr("class", t.cssClasses).attr("id", t.id).attr("data-id", t.id).attr("data-look", t.look), h = c.insert("g", ":first-child"), u = c.insert("g").attr("class", "cluster-label");
  let f = c.append("rect");
  const d = u.node().appendChild(await Pe(t.label, t.labelStyle, void 0, !0));
  let g = d.getBBox();
  if (bt(r.flowchart.htmlLabels)) {
    const R = d.children[0], O = et(d);
    g = R.getBoundingClientRect(), O.attr("width", g.width), O.attr("height", g.height);
  }
  const m = 0 * t.padding, y = m / 2, x = (t.width <= g.width + t.padding ? g.width + t.padding : t.width) + m;
  t.width <= g.width + t.padding ? t.diff = (x - t.width) / 2 - t.padding : t.diff = -t.padding;
  const b = t.height + m, k = t.height + m - g.height - 6, S = t.x - x / 2, w = t.y - b / 2;
  t.width = x;
  const C = t.y - t.height / 2 - y + g.height + 2;
  let _;
  if (t.look === "handDrawn") {
    const R = t.cssClasses.includes("statediagram-cluster-alt"), O = W.svg(c), $ = t.rx || t.ry ? O.path(me(S, w, x, b, 10), {
      roughness: 0.7,
      fill: s,
      fillStyle: "solid",
      stroke: l,
      seed: a
    }) : O.rectangle(S, w, x, b, { seed: a });
    _ = c.insert(() => $, ":first-child");
    const I = O.rectangle(S, C, x, k, {
      fill: R ? n : o,
      fillStyle: R ? "hachure" : "solid",
      stroke: l,
      seed: a
    });
    _ = c.insert(() => $, ":first-child"), f = c.insert(() => I);
  } else
    _ = h.insert("rect", ":first-child"), _.attr("class", "outer").attr("x", S).attr("y", w).attr("width", x).attr("height", b).attr("data-look", t.look), f.attr("class", "inner").attr("x", S).attr("y", C).attr("width", x).attr("height", k);
  u.attr(
    "transform",
    `translate(${t.x - g.width / 2}, ${w + 1 - (bt(r.flowchart.htmlLabels) ? 0 : 3)})`
  );
  const E = _.node().getBBox();
  return t.height = E.height, t.offsetX = 0, t.offsetY = g.height - t.padding / 2, t.labelBBox = g, t.intersect = function(R) {
    return Lr(t, R);
  }, { cluster: c, labelBBox: g };
}, "roundedWithTitle"), hw = /* @__PURE__ */ p(async (e, t) => {
  F.info("Creating subgraph rect for ", t.id, t);
  const r = at(), { themeVariables: i, handDrawnSeed: a } = r, { clusterBkg: n, clusterBorder: o } = i, { labelStyles: s, nodeStyles: l, borderStyles: c, backgroundStyles: h } = Y(t), u = e.insert("g").attr("class", "cluster " + t.cssClasses).attr("id", t.id).attr("data-look", t.look), f = bt(r.flowchart.htmlLabels), d = u.insert("g").attr("class", "cluster-label "), g = await Le(d, t.label, {
    style: t.labelStyle,
    useHtmlLabels: f,
    isNode: !0,
    width: t.width
  });
  let m = g.getBBox();
  if (bt(r.flowchart.htmlLabels)) {
    const _ = g.children[0], E = et(g);
    m = _.getBoundingClientRect(), E.attr("width", m.width), E.attr("height", m.height);
  }
  const y = t.width <= m.width + t.padding ? m.width + t.padding : t.width;
  t.width <= m.width + t.padding ? t.diff = (y - t.width) / 2 - t.padding : t.diff = -t.padding;
  const x = t.height, b = t.x - y / 2, k = t.y - x / 2;
  F.trace("Data ", t, JSON.stringify(t));
  let S;
  if (t.look === "handDrawn") {
    const _ = W.svg(u), E = H(t, {
      roughness: 0.7,
      fill: n,
      // fill: 'red',
      stroke: o,
      fillWeight: 4,
      seed: a
    }), R = _.path(me(b, k, y, x, t.rx), E);
    S = u.insert(() => (F.debug("Rough node insert CXC", R), R), ":first-child"), S.select("path:nth-child(2)").attr("style", c.join(";")), S.select("path").attr("style", h.join(";").replace("fill", "stroke"));
  } else
    S = u.insert("rect", ":first-child"), S.attr("style", l).attr("rx", t.rx).attr("ry", t.ry).attr("x", b).attr("y", k).attr("width", y).attr("height", x);
  const { subGraphTitleTopMargin: w } = vs(r);
  if (d.attr(
    "transform",
    // This puts the label on top of the box instead of inside it
    `translate(${t.x - m.width / 2}, ${t.y - t.height / 2 + w})`
  ), s) {
    const _ = d.select("span");
    _ && _.attr("style", s);
  }
  const C = S.node().getBBox();
  return t.offsetX = 0, t.width = C.width, t.height = C.height, t.offsetY = m.height - t.padding / 2, t.intersect = function(_) {
    return Lr(t, _);
  }, { cluster: u, labelBBox: m };
}, "kanbanSection"), uw = /* @__PURE__ */ p((e, t) => {
  const r = at(), { themeVariables: i, handDrawnSeed: a } = r, { nodeBorder: n } = i, o = e.insert("g").attr("class", t.cssClasses).attr("id", t.id).attr("data-look", t.look), s = o.insert("g", ":first-child"), l = 0 * t.padding, c = t.width + l;
  t.diff = -t.padding;
  const h = t.height + l, u = t.x - c / 2, f = t.y - h / 2;
  t.width = c;
  let d;
  if (t.look === "handDrawn") {
    const y = W.svg(o).rectangle(u, f, c, h, {
      fill: "lightgrey",
      roughness: 0.5,
      strokeLineDash: [5],
      stroke: n,
      seed: a
    });
    d = o.insert(() => y, ":first-child");
  } else
    d = s.insert("rect", ":first-child"), d.attr("class", "divider").attr("x", u).attr("y", f).attr("width", c).attr("height", h).attr("data-look", t.look);
  const g = d.node().getBBox();
  return t.height = g.height, t.offsetX = 0, t.offsetY = 0, t.intersect = function(m) {
    return Lr(t, m);
  }, { cluster: o, labelBBox: {} };
}, "divider"), fw = Pf, dw = {
  rect: Pf,
  squareRect: fw,
  roundedWithTitle: cw,
  noteGroup: lw,
  divider: uw,
  kanbanSection: hw
}, If = /* @__PURE__ */ new Map(), pw = /* @__PURE__ */ p(async (e, t) => {
  const r = t.shape || "rect", i = await dw[r](e, t);
  return If.set(t.id, i), i;
}, "insertCluster"), HT = /* @__PURE__ */ p(() => {
  If = /* @__PURE__ */ new Map();
}, "clear");
function Nf(e, t) {
  return e.intersect(t);
}
p(Nf, "intersectNode");
var gw = Nf;
function zf(e, t, r, i) {
  var a = e.x, n = e.y, o = a - i.x, s = n - i.y, l = Math.sqrt(t * t * s * s + r * r * o * o), c = Math.abs(t * r * o / l);
  i.x < a && (c = -c);
  var h = Math.abs(t * r * s / l);
  return i.y < n && (h = -h), { x: a + c, y: n + h };
}
p(zf, "intersectEllipse");
var qf = zf;
function Wf(e, t, r) {
  return qf(e, t, t, r);
}
p(Wf, "intersectCircle");
var mw = Wf;
function Hf(e, t, r, i) {
  var a, n, o, s, l, c, h, u, f, d, g, m, y, x, b;
  if (a = t.y - e.y, o = e.x - t.x, l = t.x * e.y - e.x * t.y, f = a * r.x + o * r.y + l, d = a * i.x + o * i.y + l, !(f !== 0 && d !== 0 && ts(f, d)) && (n = i.y - r.y, s = r.x - i.x, c = i.x * r.y - r.x * i.y, h = n * e.x + s * e.y + c, u = n * t.x + s * t.y + c, !(h !== 0 && u !== 0 && ts(h, u)) && (g = a * s - n * o, g !== 0)))
    return m = Math.abs(g / 2), y = o * c - s * l, x = y < 0 ? (y - m) / g : (y + m) / g, y = n * l - a * c, b = y < 0 ? (y - m) / g : (y + m) / g, { x, y: b };
}
p(Hf, "intersectLine");
function ts(e, t) {
  return e * t > 0;
}
p(ts, "sameSign");
var yw = Hf;
function jf(e, t, r) {
  let i = e.x, a = e.y, n = [], o = Number.POSITIVE_INFINITY, s = Number.POSITIVE_INFINITY;
  typeof t.forEach == "function" ? t.forEach(function(h) {
    o = Math.min(o, h.x), s = Math.min(s, h.y);
  }) : (o = Math.min(o, t.x), s = Math.min(s, t.y));
  let l = i - e.width / 2 - o, c = a - e.height / 2 - s;
  for (let h = 0; h < t.length; h++) {
    let u = t[h], f = t[h < t.length - 1 ? h + 1 : 0], d = yw(
      e,
      r,
      { x: l + u.x, y: c + u.y },
      { x: l + f.x, y: c + f.y }
    );
    d && n.push(d);
  }
  return n.length ? (n.length > 1 && n.sort(function(h, u) {
    let f = h.x - r.x, d = h.y - r.y, g = Math.sqrt(f * f + d * d), m = u.x - r.x, y = u.y - r.y, x = Math.sqrt(m * m + y * y);
    return g < x ? -1 : g === x ? 0 : 1;
  }), n[0]) : e;
}
p(jf, "intersectPolygon");
var xw = jf, q = {
  node: gw,
  circle: mw,
  ellipse: qf,
  polygon: xw,
  rect: Lr
};
function Yf(e, t) {
  const { labelStyles: r } = Y(t);
  t.labelStyle = r;
  const i = Z(t);
  let a = i;
  i || (a = "anchor");
  const n = e.insert("g").attr("class", a).attr("id", t.domId || t.id), o = 1, { cssStyles: s } = t, l = W.svg(n), c = H(t, { fill: "black", stroke: "none", fillStyle: "solid" });
  t.look !== "handDrawn" && (c.roughness = 0);
  const h = l.circle(0, 0, o * 2, c), u = n.insert(() => h, ":first-child");
  return u.attr("class", "anchor").attr("style", Et(s)), j(t, u), t.intersect = function(f) {
    return F.info("Circle intersect", t, o, f), q.circle(t, o, f);
  }, n;
}
p(Yf, "anchor");
function es(e, t, r, i, a, n, o) {
  const l = (e + r) / 2, c = (t + i) / 2, h = Math.atan2(i - t, r - e), u = (r - e) / 2, f = (i - t) / 2, d = u / a, g = f / n, m = Math.sqrt(d ** 2 + g ** 2);
  if (m > 1)
    throw new Error("The given radii are too small to create an arc between the points.");
  const y = Math.sqrt(1 - m ** 2), x = l + y * n * Math.sin(h) * (o ? -1 : 1), b = c - y * a * Math.cos(h) * (o ? -1 : 1), k = Math.atan2((t - b) / n, (e - x) / a);
  let w = Math.atan2((i - b) / n, (r - x) / a) - k;
  o && w < 0 && (w += 2 * Math.PI), !o && w > 0 && (w -= 2 * Math.PI);
  const C = [];
  for (let _ = 0; _ < 20; _++) {
    const E = _ / 19, R = k + E * w, O = x + a * Math.cos(R), $ = b + n * Math.sin(R);
    C.push({ x: O, y: $ });
  }
  return C;
}
p(es, "generateArcPoints");
async function Gf(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = n.width + t.padding + 20, s = n.height + t.padding, l = s / 2, c = l / (2.5 + s / 50), { cssStyles: h } = t, u = [
    { x: o / 2, y: -s / 2 },
    { x: -o / 2, y: -s / 2 },
    ...es(-o / 2, -s / 2, -o / 2, s / 2, c, l, !1),
    { x: o / 2, y: s / 2 },
    ...es(o / 2, s / 2, o / 2, -s / 2, c, l, !0)
  ], f = W.svg(a), d = H(t, {});
  t.look !== "handDrawn" && (d.roughness = 0, d.fillStyle = "solid");
  const g = rt(u), m = f.path(g, d), y = a.insert(() => m, ":first-child");
  return y.attr("class", "basic label-container"), h && t.look !== "handDrawn" && y.selectAll("path").attr("style", h), i && t.look !== "handDrawn" && y.selectAll("path").attr("style", i), y.attr("transform", `translate(${c / 2}, 0)`), j(t, y), t.intersect = function(x) {
    return q.polygon(t, u, x);
  }, a;
}
p(Gf, "bowTieRect");
function ye(e, t, r, i) {
  return e.insert("polygon", ":first-child").attr(
    "points",
    i.map(function(a) {
      return a.x + "," + a.y;
    }).join(" ")
  ).attr("class", "label-container").attr("transform", "translate(" + -t / 2 + "," + r / 2 + ")");
}
p(ye, "insertPolygonShape");
async function Uf(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = n.height + t.padding, s = 12, l = n.width + t.padding + s, c = 0, h = l, u = -o, f = 0, d = [
    { x: c + s, y: u },
    { x: h, y: u },
    { x: h, y: f },
    { x: c, y: f },
    { x: c, y: u + s },
    { x: c + s, y: u }
  ];
  let g;
  const { cssStyles: m } = t;
  if (t.look === "handDrawn") {
    const y = W.svg(a), x = H(t, {}), b = rt(d), k = y.path(b, x);
    g = a.insert(() => k, ":first-child").attr("transform", `translate(${-l / 2}, ${o / 2})`), m && g.attr("style", m);
  } else
    g = ye(a, l, o, d);
  return i && g.attr("style", i), j(t, g), t.intersect = function(y) {
    return q.polygon(t, d, y);
  }, a;
}
p(Uf, "card");
function Xf(e, t) {
  const { nodeStyles: r } = Y(t);
  t.label = "";
  const i = e.insert("g").attr("class", Z(t)).attr("id", t.domId ?? t.id), { cssStyles: a } = t, n = Math.max(28, t.width ?? 0), o = [
    { x: 0, y: n / 2 },
    { x: n / 2, y: 0 },
    { x: 0, y: -n / 2 },
    { x: -n / 2, y: 0 }
  ], s = W.svg(i), l = H(t, {});
  t.look !== "handDrawn" && (l.roughness = 0, l.fillStyle = "solid");
  const c = rt(o), h = s.path(c, l), u = i.insert(() => h, ":first-child");
  return a && t.look !== "handDrawn" && u.selectAll("path").attr("style", a), r && t.look !== "handDrawn" && u.selectAll("path").attr("style", r), t.width = 28, t.height = 28, t.intersect = function(f) {
    return q.polygon(t, o, f);
  }, i;
}
p(Xf, "choice");
async function Vf(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, halfPadding: o } = await Q(e, t, Z(t)), s = n.width / 2 + o;
  let l;
  const { cssStyles: c } = t;
  if (t.look === "handDrawn") {
    const h = W.svg(a), u = H(t, {}), f = h.circle(0, 0, s * 2, u);
    l = a.insert(() => f, ":first-child"), l.attr("class", "basic label-container").attr("style", Et(c));
  } else
    l = a.insert("circle", ":first-child").attr("class", "basic label-container").attr("style", i).attr("r", s).attr("cx", 0).attr("cy", 0);
  return j(t, l), t.intersect = function(h) {
    return F.info("Circle intersect", t, s, h), q.circle(t, s, h);
  }, a;
}
p(Vf, "circle");
function Zf(e) {
  const t = Math.cos(Math.PI / 4), r = Math.sin(Math.PI / 4), i = e * 2, a = { x: i / 2 * t, y: i / 2 * r }, n = { x: -(i / 2) * t, y: i / 2 * r }, o = { x: -(i / 2) * t, y: -(i / 2) * r }, s = { x: i / 2 * t, y: -(i / 2) * r };
  return `M ${n.x},${n.y} L ${s.x},${s.y}
                   M ${a.x},${a.y} L ${o.x},${o.y}`;
}
p(Zf, "createLine");
function Kf(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r, t.label = "";
  const a = e.insert("g").attr("class", Z(t)).attr("id", t.domId ?? t.id), n = Math.max(30, (t == null ? void 0 : t.width) ?? 0), { cssStyles: o } = t, s = W.svg(a), l = H(t, {});
  t.look !== "handDrawn" && (l.roughness = 0, l.fillStyle = "solid");
  const c = s.circle(0, 0, n * 2, l), h = Zf(n), u = s.path(h, l), f = a.insert(() => c, ":first-child");
  return f.insert(() => u), o && t.look !== "handDrawn" && f.selectAll("path").attr("style", o), i && t.look !== "handDrawn" && f.selectAll("path").attr("style", i), j(t, f), t.intersect = function(d) {
    return F.info("crossedCircle intersect", t, { radius: n, point: d }), q.circle(t, n, d);
  }, a;
}
p(Kf, "crossedCircle");
function oe(e, t, r, i = 100, a = 0, n = 180) {
  const o = [], s = a * Math.PI / 180, h = (n * Math.PI / 180 - s) / (i - 1);
  for (let u = 0; u < i; u++) {
    const f = s + u * h, d = e + r * Math.cos(f), g = t + r * Math.sin(f);
    o.push({ x: -d, y: -g });
  }
  return o;
}
p(oe, "generateCirclePoints");
async function Qf(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = n.width + (t.padding ?? 0), l = n.height + (t.padding ?? 0), c = Math.max(5, l * 0.1), { cssStyles: h } = t, u = [
    ...oe(s / 2, -l / 2, c, 30, -90, 0),
    { x: -s / 2 - c, y: c },
    ...oe(s / 2 + c * 2, -c, c, 20, -180, -270),
    ...oe(s / 2 + c * 2, c, c, 20, -90, -180),
    { x: -s / 2 - c, y: -l / 2 },
    ...oe(s / 2, l / 2, c, 20, 0, 90)
  ], f = [
    { x: s / 2, y: -l / 2 - c },
    { x: -s / 2, y: -l / 2 - c },
    ...oe(s / 2, -l / 2, c, 20, -90, 0),
    { x: -s / 2 - c, y: -c },
    ...oe(s / 2 + s * 0.1, -c, c, 20, -180, -270),
    ...oe(s / 2 + s * 0.1, c, c, 20, -90, -180),
    { x: -s / 2 - c, y: l / 2 },
    ...oe(s / 2, l / 2, c, 20, 0, 90),
    { x: -s / 2, y: l / 2 + c },
    { x: s / 2, y: l / 2 + c }
  ], d = W.svg(a), g = H(t, { fill: "none" });
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const y = rt(u).replace("Z", ""), x = d.path(y, g), b = rt(f), k = d.path(b, { ...g }), S = a.insert("g", ":first-child");
  return S.insert(() => k, ":first-child").attr("stroke-opacity", 0), S.insert(() => x, ":first-child"), S.attr("class", "text"), h && t.look !== "handDrawn" && S.selectAll("path").attr("style", h), i && t.look !== "handDrawn" && S.selectAll("path").attr("style", i), S.attr("transform", `translate(${c}, 0)`), o.attr(
    "transform",
    `translate(${-s / 2 + c - (n.x - (n.left ?? 0))},${-l / 2 + (t.padding ?? 0) / 2 - (n.y - (n.top ?? 0))})`
  ), j(t, S), t.intersect = function(w) {
    return q.polygon(t, f, w);
  }, a;
}
p(Qf, "curlyBraceLeft");
function le(e, t, r, i = 100, a = 0, n = 180) {
  const o = [], s = a * Math.PI / 180, h = (n * Math.PI / 180 - s) / (i - 1);
  for (let u = 0; u < i; u++) {
    const f = s + u * h, d = e + r * Math.cos(f), g = t + r * Math.sin(f);
    o.push({ x: d, y: g });
  }
  return o;
}
p(le, "generateCirclePoints");
async function Jf(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = n.width + (t.padding ?? 0), l = n.height + (t.padding ?? 0), c = Math.max(5, l * 0.1), { cssStyles: h } = t, u = [
    ...le(s / 2, -l / 2, c, 20, -90, 0),
    { x: s / 2 + c, y: -c },
    ...le(s / 2 + c * 2, -c, c, 20, -180, -270),
    ...le(s / 2 + c * 2, c, c, 20, -90, -180),
    { x: s / 2 + c, y: l / 2 },
    ...le(s / 2, l / 2, c, 20, 0, 90)
  ], f = [
    { x: -s / 2, y: -l / 2 - c },
    { x: s / 2, y: -l / 2 - c },
    ...le(s / 2, -l / 2, c, 20, -90, 0),
    { x: s / 2 + c, y: -c },
    ...le(s / 2 + c * 2, -c, c, 20, -180, -270),
    ...le(s / 2 + c * 2, c, c, 20, -90, -180),
    { x: s / 2 + c, y: l / 2 },
    ...le(s / 2, l / 2, c, 20, 0, 90),
    { x: s / 2, y: l / 2 + c },
    { x: -s / 2, y: l / 2 + c }
  ], d = W.svg(a), g = H(t, { fill: "none" });
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const y = rt(u).replace("Z", ""), x = d.path(y, g), b = rt(f), k = d.path(b, { ...g }), S = a.insert("g", ":first-child");
  return S.insert(() => k, ":first-child").attr("stroke-opacity", 0), S.insert(() => x, ":first-child"), S.attr("class", "text"), h && t.look !== "handDrawn" && S.selectAll("path").attr("style", h), i && t.look !== "handDrawn" && S.selectAll("path").attr("style", i), S.attr("transform", `translate(${-c}, 0)`), o.attr(
    "transform",
    `translate(${-s / 2 + (t.padding ?? 0) / 2 - (n.x - (n.left ?? 0))},${-l / 2 + (t.padding ?? 0) / 2 - (n.y - (n.top ?? 0))})`
  ), j(t, S), t.intersect = function(w) {
    return q.polygon(t, f, w);
  }, a;
}
p(Jf, "curlyBraceRight");
function wt(e, t, r, i = 100, a = 0, n = 180) {
  const o = [], s = a * Math.PI / 180, h = (n * Math.PI / 180 - s) / (i - 1);
  for (let u = 0; u < i; u++) {
    const f = s + u * h, d = e + r * Math.cos(f), g = t + r * Math.sin(f);
    o.push({ x: -d, y: -g });
  }
  return o;
}
p(wt, "generateCirclePoints");
async function td(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = n.width + (t.padding ?? 0), l = n.height + (t.padding ?? 0), c = Math.max(5, l * 0.1), { cssStyles: h } = t, u = [
    ...wt(s / 2, -l / 2, c, 30, -90, 0),
    { x: -s / 2 - c, y: c },
    ...wt(s / 2 + c * 2, -c, c, 20, -180, -270),
    ...wt(s / 2 + c * 2, c, c, 20, -90, -180),
    { x: -s / 2 - c, y: -l / 2 },
    ...wt(s / 2, l / 2, c, 20, 0, 90)
  ], f = [
    ...wt(-s / 2 + c + c / 2, -l / 2, c, 20, -90, -180),
    { x: s / 2 - c / 2, y: c },
    ...wt(-s / 2 - c / 2, -c, c, 20, 0, 90),
    ...wt(-s / 2 - c / 2, c, c, 20, -90, 0),
    { x: s / 2 - c / 2, y: -c },
    ...wt(-s / 2 + c + c / 2, l / 2, c, 30, -180, -270)
  ], d = [
    { x: s / 2, y: -l / 2 - c },
    { x: -s / 2, y: -l / 2 - c },
    ...wt(s / 2, -l / 2, c, 20, -90, 0),
    { x: -s / 2 - c, y: -c },
    ...wt(s / 2 + c * 2, -c, c, 20, -180, -270),
    ...wt(s / 2 + c * 2, c, c, 20, -90, -180),
    { x: -s / 2 - c, y: l / 2 },
    ...wt(s / 2, l / 2, c, 20, 0, 90),
    { x: -s / 2, y: l / 2 + c },
    { x: s / 2 - c - c / 2, y: l / 2 + c },
    ...wt(-s / 2 + c + c / 2, -l / 2, c, 20, -90, -180),
    { x: s / 2 - c / 2, y: c },
    ...wt(-s / 2 - c / 2, -c, c, 20, 0, 90),
    ...wt(-s / 2 - c / 2, c, c, 20, -90, 0),
    { x: s / 2 - c / 2, y: -c },
    ...wt(-s / 2 + c + c / 2, l / 2, c, 30, -180, -270)
  ], g = W.svg(a), m = H(t, { fill: "none" });
  t.look !== "handDrawn" && (m.roughness = 0, m.fillStyle = "solid");
  const x = rt(u).replace("Z", ""), b = g.path(x, m), S = rt(f).replace("Z", ""), w = g.path(S, m), C = rt(d), _ = g.path(C, { ...m }), E = a.insert("g", ":first-child");
  return E.insert(() => _, ":first-child").attr("stroke-opacity", 0), E.insert(() => b, ":first-child"), E.insert(() => w, ":first-child"), E.attr("class", "text"), h && t.look !== "handDrawn" && E.selectAll("path").attr("style", h), i && t.look !== "handDrawn" && E.selectAll("path").attr("style", i), E.attr("transform", `translate(${c - c / 4}, 0)`), o.attr(
    "transform",
    `translate(${-s / 2 + (t.padding ?? 0) / 2 - (n.x - (n.left ?? 0))},${-l / 2 + (t.padding ?? 0) / 2 - (n.y - (n.top ?? 0))})`
  ), j(t, E), t.intersect = function(R) {
    return q.polygon(t, d, R);
  }, a;
}
p(td, "curlyBraces");
async function ed(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = 80, s = 20, l = Math.max(o, (n.width + (t.padding ?? 0) * 2) * 1.25, (t == null ? void 0 : t.width) ?? 0), c = Math.max(s, n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), h = c / 2, { cssStyles: u } = t, f = W.svg(a), d = H(t, {});
  t.look !== "handDrawn" && (d.roughness = 0, d.fillStyle = "solid");
  const g = l, m = c, y = g - h, x = m / 4, b = [
    { x: y, y: 0 },
    { x, y: 0 },
    { x: 0, y: m / 2 },
    { x, y: m },
    { x: y, y: m },
    ...so(-y, -m / 2, h, 50, 270, 90)
  ], k = rt(b), S = f.path(k, d), w = a.insert(() => S, ":first-child");
  return w.attr("class", "basic label-container"), u && t.look !== "handDrawn" && w.selectChildren("path").attr("style", u), i && t.look !== "handDrawn" && w.selectChildren("path").attr("style", i), w.attr("transform", `translate(${-l / 2}, ${-c / 2})`), j(t, w), t.intersect = function(C) {
    return q.polygon(t, b, C);
  }, a;
}
p(ed, "curvedTrapezoid");
var bw = /* @__PURE__ */ p((e, t, r, i, a, n) => [
  `M${e},${t + n}`,
  `a${a},${n} 0,0,0 ${r},0`,
  `a${a},${n} 0,0,0 ${-r},0`,
  `l0,${i}`,
  `a${a},${n} 0,0,0 ${r},0`,
  `l0,${-i}`
].join(" "), "createCylinderPathD"), Cw = /* @__PURE__ */ p((e, t, r, i, a, n) => [
  `M${e},${t + n}`,
  `M${e + r},${t + n}`,
  `a${a},${n} 0,0,0 ${-r},0`,
  `l0,${i}`,
  `a${a},${n} 0,0,0 ${r},0`,
  `l0,${-i}`
].join(" "), "createOuterCylinderPathD"), kw = /* @__PURE__ */ p((e, t, r, i, a, n) => [`M${e - r / 2},${-i / 2}`, `a${a},${n} 0,0,0 ${r},0`].join(" "), "createInnerCylinderPathD");
async function rd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + t.padding, t.width ?? 0), l = s / 2, c = l / (2.5 + s / 50), h = Math.max(n.height + c + t.padding, t.height ?? 0);
  let u;
  const { cssStyles: f } = t;
  if (t.look === "handDrawn") {
    const d = W.svg(a), g = Cw(0, 0, s, h, l, c), m = kw(0, c, s, h, l, c), y = d.path(g, H(t, {})), x = d.path(m, H(t, { fill: "none" }));
    u = a.insert(() => x, ":first-child"), u = a.insert(() => y, ":first-child"), u.attr("class", "basic label-container"), f && u.attr("style", f);
  } else {
    const d = bw(0, 0, s, h, l, c);
    u = a.insert("path", ":first-child").attr("d", d).attr("class", "basic label-container").attr("style", Et(f)).attr("style", i);
  }
  return u.attr("label-offset-y", c), u.attr("transform", `translate(${-s / 2}, ${-(h / 2 + c)})`), j(t, u), o.attr(
    "transform",
    `translate(${-(n.width / 2) - (n.x - (n.left ?? 0))}, ${-(n.height / 2) + (t.padding ?? 0) / 1.5 - (n.y - (n.top ?? 0))})`
  ), t.intersect = function(d) {
    const g = q.rect(t, d), m = g.x - (t.x ?? 0);
    if (l != 0 && (Math.abs(m) < (t.width ?? 0) / 2 || Math.abs(m) == (t.width ?? 0) / 2 && Math.abs(g.y - (t.y ?? 0)) > (t.height ?? 0) / 2 - c)) {
      let y = c * c * (1 - m * m / (l * l));
      y > 0 && (y = Math.sqrt(y)), y = c - y, d.y - (t.y ?? 0) > 0 && (y = -y), g.y += y;
    }
    return g;
  }, a;
}
p(rd, "cylinder");
async function id(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = n.width + t.padding, l = n.height + t.padding, c = l * 0.2, h = -s / 2, u = -l / 2 - c / 2, { cssStyles: f } = t, d = W.svg(a), g = H(t, {});
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const m = [
    { x: h, y: u + c },
    { x: -h, y: u + c },
    { x: -h, y: -u },
    { x: h, y: -u },
    { x: h, y: u },
    { x: -h, y: u },
    { x: -h, y: u + c }
  ], y = d.polygon(
    m.map((b) => [b.x, b.y]),
    g
  ), x = a.insert(() => y, ":first-child");
  return x.attr("class", "basic label-container"), f && t.look !== "handDrawn" && x.selectAll("path").attr("style", f), i && t.look !== "handDrawn" && x.selectAll("path").attr("style", i), o.attr(
    "transform",
    `translate(${h + (t.padding ?? 0) / 2 - (n.x - (n.left ?? 0))}, ${u + c + (t.padding ?? 0) / 2 - (n.y - (n.top ?? 0))})`
  ), j(t, x), t.intersect = function(b) {
    return q.rect(t, b);
  }, a;
}
p(id, "dividedRectangle");
async function ad(e, t) {
  var f, d;
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, halfPadding: o } = await Q(e, t, Z(t)), l = n.width / 2 + o + 5, c = n.width / 2 + o;
  let h;
  const { cssStyles: u } = t;
  if (t.look === "handDrawn") {
    const g = W.svg(a), m = H(t, { roughness: 0.2, strokeWidth: 2.5 }), y = H(t, { roughness: 0.2, strokeWidth: 1.5 }), x = g.circle(0, 0, l * 2, m), b = g.circle(0, 0, c * 2, y);
    h = a.insert("g", ":first-child"), h.attr("class", Et(t.cssClasses)).attr("style", Et(u)), (f = h.node()) == null || f.appendChild(x), (d = h.node()) == null || d.appendChild(b);
  } else {
    h = a.insert("g", ":first-child");
    const g = h.insert("circle", ":first-child"), m = h.insert("circle");
    h.attr("class", "basic label-container").attr("style", i), g.attr("class", "outer-circle").attr("style", i).attr("r", l).attr("cx", 0).attr("cy", 0), m.attr("class", "inner-circle").attr("style", i).attr("r", c).attr("cx", 0).attr("cy", 0);
  }
  return j(t, h), t.intersect = function(g) {
    return F.info("DoubleCircle intersect", t, l, g), q.circle(t, l, g);
  }, a;
}
p(ad, "doublecircle");
function nd(e, t, { config: { themeVariables: r } }) {
  const { labelStyles: i, nodeStyles: a } = Y(t);
  t.label = "", t.labelStyle = i;
  const n = e.insert("g").attr("class", Z(t)).attr("id", t.domId ?? t.id), o = 7, { cssStyles: s } = t, l = W.svg(n), { nodeBorder: c } = r, h = H(t, { fillStyle: "solid" });
  t.look !== "handDrawn" && (h.roughness = 0);
  const u = l.circle(0, 0, o * 2, h), f = n.insert(() => u, ":first-child");
  return f.selectAll("path").attr("style", `fill: ${c} !important;`), s && s.length > 0 && t.look !== "handDrawn" && f.selectAll("path").attr("style", s), a && t.look !== "handDrawn" && f.selectAll("path").attr("style", a), j(t, f), t.intersect = function(d) {
    return F.info("filledCircle intersect", t, { radius: o, point: d }), q.circle(t, o, d);
  }, n;
}
p(nd, "filledCircle");
async function sd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = n.width + (t.padding ?? 0), l = s + n.height, c = s + n.height, h = [
    { x: 0, y: -l },
    { x: c, y: -l },
    { x: c / 2, y: 0 }
  ], { cssStyles: u } = t, f = W.svg(a), d = H(t, {});
  t.look !== "handDrawn" && (d.roughness = 0, d.fillStyle = "solid");
  const g = rt(h), m = f.path(g, d), y = a.insert(() => m, ":first-child").attr("transform", `translate(${-l / 2}, ${l / 2})`);
  return u && t.look !== "handDrawn" && y.selectChildren("path").attr("style", u), i && t.look !== "handDrawn" && y.selectChildren("path").attr("style", i), t.width = s, t.height = l, j(t, y), o.attr(
    "transform",
    `translate(${-n.width / 2 - (n.x - (n.left ?? 0))}, ${-l / 2 + (t.padding ?? 0) / 2 + (n.y - (n.top ?? 0))})`
  ), t.intersect = function(x) {
    return F.info("Triangle intersect", t, h, x), q.polygon(t, h, x);
  }, a;
}
p(sd, "flippedTriangle");
function od(e, t, { dir: r, config: { state: i, themeVariables: a } }) {
  const { nodeStyles: n } = Y(t);
  t.label = "";
  const o = e.insert("g").attr("class", Z(t)).attr("id", t.domId ?? t.id), { cssStyles: s } = t;
  let l = Math.max(70, (t == null ? void 0 : t.width) ?? 0), c = Math.max(10, (t == null ? void 0 : t.height) ?? 0);
  r === "LR" && (l = Math.max(10, (t == null ? void 0 : t.width) ?? 0), c = Math.max(70, (t == null ? void 0 : t.height) ?? 0));
  const h = -1 * l / 2, u = -1 * c / 2, f = W.svg(o), d = H(t, {
    stroke: a.lineColor,
    fill: a.lineColor
  });
  t.look !== "handDrawn" && (d.roughness = 0, d.fillStyle = "solid");
  const g = f.rectangle(h, u, l, c, d), m = o.insert(() => g, ":first-child");
  s && t.look !== "handDrawn" && m.selectAll("path").attr("style", s), n && t.look !== "handDrawn" && m.selectAll("path").attr("style", n), j(t, m);
  const y = (i == null ? void 0 : i.padding) ?? 0;
  return t.width && t.height && (t.width += y / 2 || 0, t.height += y / 2 || 0), t.intersect = function(x) {
    return q.rect(t, x);
  }, o;
}
p(od, "forkJoin");
async function ld(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const a = 80, n = 50, { shapeSvg: o, bbox: s } = await Q(e, t, Z(t)), l = Math.max(a, s.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), c = Math.max(n, s.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), h = c / 2, { cssStyles: u } = t, f = W.svg(o), d = H(t, {});
  t.look !== "handDrawn" && (d.roughness = 0, d.fillStyle = "solid");
  const g = [
    { x: -l / 2, y: -c / 2 },
    { x: l / 2 - h, y: -c / 2 },
    ...so(-l / 2 + h, 0, h, 50, 90, 270),
    { x: l / 2 - h, y: c / 2 },
    { x: -l / 2, y: c / 2 }
  ], m = rt(g), y = f.path(m, d), x = o.insert(() => y, ":first-child");
  return x.attr("class", "basic label-container"), u && t.look !== "handDrawn" && x.selectChildren("path").attr("style", u), i && t.look !== "handDrawn" && x.selectChildren("path").attr("style", i), j(t, x), t.intersect = function(b) {
    return F.info("Pill intersect", t, { radius: h, point: b }), q.polygon(t, g, b);
  }, o;
}
p(ld, "halfRoundedRectangle");
var ww = /* @__PURE__ */ p((e, t, r, i, a) => [
  `M${e + a},${t}`,
  `L${e + r - a},${t}`,
  `L${e + r},${t - i / 2}`,
  `L${e + r - a},${t - i}`,
  `L${e + a},${t - i}`,
  `L${e},${t - i / 2}`,
  "Z"
].join(" "), "createHexagonPathD");
async function cd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = 4, s = n.height + t.padding, l = s / o, c = n.width + 2 * l + t.padding, h = [
    { x: l, y: 0 },
    { x: c - l, y: 0 },
    { x: c, y: -s / 2 },
    { x: c - l, y: -s },
    { x: l, y: -s },
    { x: 0, y: -s / 2 }
  ];
  let u;
  const { cssStyles: f } = t;
  if (t.look === "handDrawn") {
    const d = W.svg(a), g = H(t, {}), m = ww(0, 0, c, s, l), y = d.path(m, g);
    u = a.insert(() => y, ":first-child").attr("transform", `translate(${-c / 2}, ${s / 2})`), f && u.attr("style", f);
  } else
    u = ye(a, c, s, h);
  return i && u.attr("style", i), t.width = c, t.height = s, j(t, u), t.intersect = function(d) {
    return q.polygon(t, h, d);
  }, a;
}
p(cd, "hexagon");
async function hd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.label = "", t.labelStyle = r;
  const { shapeSvg: a } = await Q(e, t, Z(t)), n = Math.max(30, (t == null ? void 0 : t.width) ?? 0), o = Math.max(30, (t == null ? void 0 : t.height) ?? 0), { cssStyles: s } = t, l = W.svg(a), c = H(t, {});
  t.look !== "handDrawn" && (c.roughness = 0, c.fillStyle = "solid");
  const h = [
    { x: 0, y: 0 },
    { x: n, y: 0 },
    { x: 0, y: o },
    { x: n, y: o }
  ], u = rt(h), f = l.path(u, c), d = a.insert(() => f, ":first-child");
  return d.attr("class", "basic label-container"), s && t.look !== "handDrawn" && d.selectChildren("path").attr("style", s), i && t.look !== "handDrawn" && d.selectChildren("path").attr("style", i), d.attr("transform", `translate(${-n / 2}, ${-o / 2})`), j(t, d), t.intersect = function(g) {
    return F.info("Pill intersect", t, { points: h }), q.polygon(t, h, g);
  }, a;
}
p(hd, "hourglass");
async function ud(e, t, { config: { themeVariables: r, flowchart: i } }) {
  const { labelStyles: a } = Y(t);
  t.labelStyle = a;
  const n = t.assetHeight ?? 48, o = t.assetWidth ?? 48, s = Math.max(n, o), l = i == null ? void 0 : i.wrappingWidth;
  t.width = Math.max(s, l ?? 0);
  const { shapeSvg: c, bbox: h, label: u } = await Q(e, t, "icon-shape default"), f = t.pos === "t", d = s, g = s, { nodeBorder: m } = r, { stylesMap: y } = Sr(t), x = -g / 2, b = -d / 2, k = t.label ? 8 : 0, S = W.svg(c), w = H(t, { stroke: "none", fill: "none" });
  t.look !== "handDrawn" && (w.roughness = 0, w.fillStyle = "solid");
  const C = S.rectangle(x, b, g, d, w), _ = Math.max(g, h.width), E = d + h.height + k, R = S.rectangle(-_ / 2, -E / 2, _, E, {
    ...w,
    fill: "transparent",
    stroke: "none"
  }), O = c.insert(() => C, ":first-child"), $ = c.insert(() => R);
  if (t.icon) {
    const I = c.append("g");
    I.html(
      `<g>${await yi(t.icon, {
        height: s,
        width: s,
        fallbackPrefix: ""
      })}</g>`
    );
    const D = I.node().getBBox(), B = D.width, M = D.height, T = D.x, A = D.y;
    I.attr(
      "transform",
      `translate(${-B / 2 - T},${f ? h.height / 2 + k / 2 - M / 2 - A : -h.height / 2 - k / 2 - M / 2 - A})`
    ), I.attr("style", `color: ${y.get("stroke") ?? m};`);
  }
  return u.attr(
    "transform",
    `translate(${-h.width / 2 - (h.x - (h.left ?? 0))},${f ? -E / 2 : E / 2 - h.height})`
  ), O.attr(
    "transform",
    `translate(0,${f ? h.height / 2 + k / 2 : -h.height / 2 - k / 2})`
  ), j(t, $), t.intersect = function(I) {
    if (F.info("iconSquare intersect", t, I), !t.label)
      return q.rect(t, I);
    const D = t.x ?? 0, B = t.y ?? 0, M = t.height ?? 0;
    let T = [];
    return f ? T = [
      { x: D - h.width / 2, y: B - M / 2 },
      { x: D + h.width / 2, y: B - M / 2 },
      { x: D + h.width / 2, y: B - M / 2 + h.height + k },
      { x: D + g / 2, y: B - M / 2 + h.height + k },
      { x: D + g / 2, y: B + M / 2 },
      { x: D - g / 2, y: B + M / 2 },
      { x: D - g / 2, y: B - M / 2 + h.height + k },
      { x: D - h.width / 2, y: B - M / 2 + h.height + k }
    ] : T = [
      { x: D - g / 2, y: B - M / 2 },
      { x: D + g / 2, y: B - M / 2 },
      { x: D + g / 2, y: B - M / 2 + d },
      { x: D + h.width / 2, y: B - M / 2 + d },
      { x: D + h.width / 2 / 2, y: B + M / 2 },
      { x: D - h.width / 2, y: B + M / 2 },
      { x: D - h.width / 2, y: B - M / 2 + d },
      { x: D - g / 2, y: B - M / 2 + d }
    ], q.polygon(t, T, I);
  }, c;
}
p(ud, "icon");
async function fd(e, t, { config: { themeVariables: r, flowchart: i } }) {
  const { labelStyles: a } = Y(t);
  t.labelStyle = a;
  const n = t.assetHeight ?? 48, o = t.assetWidth ?? 48, s = Math.max(n, o), l = i == null ? void 0 : i.wrappingWidth;
  t.width = Math.max(s, l ?? 0);
  const { shapeSvg: c, bbox: h, label: u } = await Q(e, t, "icon-shape default"), f = 20, d = t.label ? 8 : 0, g = t.pos === "t", { nodeBorder: m, mainBkg: y } = r, { stylesMap: x } = Sr(t), b = W.svg(c), k = H(t, {});
  t.look !== "handDrawn" && (k.roughness = 0, k.fillStyle = "solid");
  const S = x.get("fill");
  k.stroke = S ?? y;
  const w = c.append("g");
  t.icon && w.html(
    `<g>${await yi(t.icon, {
      height: s,
      width: s,
      fallbackPrefix: ""
    })}</g>`
  );
  const C = w.node().getBBox(), _ = C.width, E = C.height, R = C.x, O = C.y, $ = Math.max(_, E) * Math.SQRT2 + f * 2, I = b.circle(0, 0, $, k), D = Math.max($, h.width), B = $ + h.height + d, M = b.rectangle(-D / 2, -B / 2, D, B, {
    ...k,
    fill: "transparent",
    stroke: "none"
  }), T = c.insert(() => I, ":first-child"), A = c.insert(() => M);
  return w.attr(
    "transform",
    `translate(${-_ / 2 - R},${g ? h.height / 2 + d / 2 - E / 2 - O : -h.height / 2 - d / 2 - E / 2 - O})`
  ), w.attr("style", `color: ${x.get("stroke") ?? m};`), u.attr(
    "transform",
    `translate(${-h.width / 2 - (h.x - (h.left ?? 0))},${g ? -B / 2 : B / 2 - h.height})`
  ), T.attr(
    "transform",
    `translate(0,${g ? h.height / 2 + d / 2 : -h.height / 2 - d / 2})`
  ), j(t, A), t.intersect = function(L) {
    return F.info("iconSquare intersect", t, L), q.rect(t, L);
  }, c;
}
p(fd, "iconCircle");
async function dd(e, t, { config: { themeVariables: r, flowchart: i } }) {
  const { labelStyles: a } = Y(t);
  t.labelStyle = a;
  const n = t.assetHeight ?? 48, o = t.assetWidth ?? 48, s = Math.max(n, o), l = i == null ? void 0 : i.wrappingWidth;
  t.width = Math.max(s, l ?? 0);
  const { shapeSvg: c, bbox: h, halfPadding: u, label: f } = await Q(
    e,
    t,
    "icon-shape default"
  ), d = t.pos === "t", g = s + u * 2, m = s + u * 2, { nodeBorder: y, mainBkg: x } = r, { stylesMap: b } = Sr(t), k = -m / 2, S = -g / 2, w = t.label ? 8 : 0, C = W.svg(c), _ = H(t, {});
  t.look !== "handDrawn" && (_.roughness = 0, _.fillStyle = "solid");
  const E = b.get("fill");
  _.stroke = E ?? x;
  const R = C.path(me(k, S, m, g, 5), _), O = Math.max(m, h.width), $ = g + h.height + w, I = C.rectangle(-O / 2, -$ / 2, O, $, {
    ..._,
    fill: "transparent",
    stroke: "none"
  }), D = c.insert(() => R, ":first-child").attr("class", "icon-shape2"), B = c.insert(() => I);
  if (t.icon) {
    const M = c.append("g");
    M.html(
      `<g>${await yi(t.icon, {
        height: s,
        width: s,
        fallbackPrefix: ""
      })}</g>`
    );
    const T = M.node().getBBox(), A = T.width, L = T.height, N = T.x, U = T.y;
    M.attr(
      "transform",
      `translate(${-A / 2 - N},${d ? h.height / 2 + w / 2 - L / 2 - U : -h.height / 2 - w / 2 - L / 2 - U})`
    ), M.attr("style", `color: ${b.get("stroke") ?? y};`);
  }
  return f.attr(
    "transform",
    `translate(${-h.width / 2 - (h.x - (h.left ?? 0))},${d ? -$ / 2 : $ / 2 - h.height})`
  ), D.attr(
    "transform",
    `translate(0,${d ? h.height / 2 + w / 2 : -h.height / 2 - w / 2})`
  ), j(t, B), t.intersect = function(M) {
    if (F.info("iconSquare intersect", t, M), !t.label)
      return q.rect(t, M);
    const T = t.x ?? 0, A = t.y ?? 0, L = t.height ?? 0;
    let N = [];
    return d ? N = [
      { x: T - h.width / 2, y: A - L / 2 },
      { x: T + h.width / 2, y: A - L / 2 },
      { x: T + h.width / 2, y: A - L / 2 + h.height + w },
      { x: T + m / 2, y: A - L / 2 + h.height + w },
      { x: T + m / 2, y: A + L / 2 },
      { x: T - m / 2, y: A + L / 2 },
      { x: T - m / 2, y: A - L / 2 + h.height + w },
      { x: T - h.width / 2, y: A - L / 2 + h.height + w }
    ] : N = [
      { x: T - m / 2, y: A - L / 2 },
      { x: T + m / 2, y: A - L / 2 },
      { x: T + m / 2, y: A - L / 2 + g },
      { x: T + h.width / 2, y: A - L / 2 + g },
      { x: T + h.width / 2 / 2, y: A + L / 2 },
      { x: T - h.width / 2, y: A + L / 2 },
      { x: T - h.width / 2, y: A - L / 2 + g },
      { x: T - m / 2, y: A - L / 2 + g }
    ], q.polygon(t, N, M);
  }, c;
}
p(dd, "iconRounded");
async function pd(e, t, { config: { themeVariables: r, flowchart: i } }) {
  const { labelStyles: a } = Y(t);
  t.labelStyle = a;
  const n = t.assetHeight ?? 48, o = t.assetWidth ?? 48, s = Math.max(n, o), l = i == null ? void 0 : i.wrappingWidth;
  t.width = Math.max(s, l ?? 0);
  const { shapeSvg: c, bbox: h, halfPadding: u, label: f } = await Q(
    e,
    t,
    "icon-shape default"
  ), d = t.pos === "t", g = s + u * 2, m = s + u * 2, { nodeBorder: y, mainBkg: x } = r, { stylesMap: b } = Sr(t), k = -m / 2, S = -g / 2, w = t.label ? 8 : 0, C = W.svg(c), _ = H(t, {});
  t.look !== "handDrawn" && (_.roughness = 0, _.fillStyle = "solid");
  const E = b.get("fill");
  _.stroke = E ?? x;
  const R = C.path(me(k, S, m, g, 0.1), _), O = Math.max(m, h.width), $ = g + h.height + w, I = C.rectangle(-O / 2, -$ / 2, O, $, {
    ..._,
    fill: "transparent",
    stroke: "none"
  }), D = c.insert(() => R, ":first-child"), B = c.insert(() => I);
  if (t.icon) {
    const M = c.append("g");
    M.html(
      `<g>${await yi(t.icon, {
        height: s,
        width: s,
        fallbackPrefix: ""
      })}</g>`
    );
    const T = M.node().getBBox(), A = T.width, L = T.height, N = T.x, U = T.y;
    M.attr(
      "transform",
      `translate(${-A / 2 - N},${d ? h.height / 2 + w / 2 - L / 2 - U : -h.height / 2 - w / 2 - L / 2 - U})`
    ), M.attr("style", `color: ${b.get("stroke") ?? y};`);
  }
  return f.attr(
    "transform",
    `translate(${-h.width / 2 - (h.x - (h.left ?? 0))},${d ? -$ / 2 : $ / 2 - h.height})`
  ), D.attr(
    "transform",
    `translate(0,${d ? h.height / 2 + w / 2 : -h.height / 2 - w / 2})`
  ), j(t, B), t.intersect = function(M) {
    if (F.info("iconSquare intersect", t, M), !t.label)
      return q.rect(t, M);
    const T = t.x ?? 0, A = t.y ?? 0, L = t.height ?? 0;
    let N = [];
    return d ? N = [
      { x: T - h.width / 2, y: A - L / 2 },
      { x: T + h.width / 2, y: A - L / 2 },
      { x: T + h.width / 2, y: A - L / 2 + h.height + w },
      { x: T + m / 2, y: A - L / 2 + h.height + w },
      { x: T + m / 2, y: A + L / 2 },
      { x: T - m / 2, y: A + L / 2 },
      { x: T - m / 2, y: A - L / 2 + h.height + w },
      { x: T - h.width / 2, y: A - L / 2 + h.height + w }
    ] : N = [
      { x: T - m / 2, y: A - L / 2 },
      { x: T + m / 2, y: A - L / 2 },
      { x: T + m / 2, y: A - L / 2 + g },
      { x: T + h.width / 2, y: A - L / 2 + g },
      { x: T + h.width / 2 / 2, y: A + L / 2 },
      { x: T - h.width / 2, y: A + L / 2 },
      { x: T - h.width / 2, y: A - L / 2 + g },
      { x: T - m / 2, y: A - L / 2 + g }
    ], q.polygon(t, N, M);
  }, c;
}
p(pd, "iconSquare");
async function gd(e, t, { config: { flowchart: r } }) {
  const i = new Image();
  i.src = (t == null ? void 0 : t.img) ?? "", await i.decode();
  const a = Number(i.naturalWidth.toString().replace("px", "")), n = Number(i.naturalHeight.toString().replace("px", ""));
  t.imageAspectRatio = a / n;
  const { labelStyles: o } = Y(t);
  t.labelStyle = o;
  const s = r == null ? void 0 : r.wrappingWidth;
  t.defaultWidth = r == null ? void 0 : r.wrappingWidth;
  const l = Math.max(
    t.label ? s ?? 0 : 0,
    (t == null ? void 0 : t.assetWidth) ?? a
  ), c = t.constraint === "on" && t != null && t.assetHeight ? t.assetHeight * t.imageAspectRatio : l, h = t.constraint === "on" ? c / t.imageAspectRatio : (t == null ? void 0 : t.assetHeight) ?? n;
  t.width = Math.max(c, s ?? 0);
  const { shapeSvg: u, bbox: f, label: d } = await Q(e, t, "image-shape default"), g = t.pos === "t", m = -c / 2, y = -h / 2, x = t.label ? 8 : 0, b = W.svg(u), k = H(t, {});
  t.look !== "handDrawn" && (k.roughness = 0, k.fillStyle = "solid");
  const S = b.rectangle(m, y, c, h, k), w = Math.max(c, f.width), C = h + f.height + x, _ = b.rectangle(-w / 2, -C / 2, w, C, {
    ...k,
    fill: "none",
    stroke: "none"
  }), E = u.insert(() => S, ":first-child"), R = u.insert(() => _);
  if (t.img) {
    const O = u.append("image");
    O.attr("href", t.img), O.attr("width", c), O.attr("height", h), O.attr("preserveAspectRatio", "none"), O.attr(
      "transform",
      `translate(${-c / 2},${g ? C / 2 - h : -C / 2})`
    );
  }
  return d.attr(
    "transform",
    `translate(${-f.width / 2 - (f.x - (f.left ?? 0))},${g ? -h / 2 - f.height / 2 - x / 2 : h / 2 - f.height / 2 + x / 2})`
  ), E.attr(
    "transform",
    `translate(0,${g ? f.height / 2 + x / 2 : -f.height / 2 - x / 2})`
  ), j(t, R), t.intersect = function(O) {
    if (F.info("iconSquare intersect", t, O), !t.label)
      return q.rect(t, O);
    const $ = t.x ?? 0, I = t.y ?? 0, D = t.height ?? 0;
    let B = [];
    return g ? B = [
      { x: $ - f.width / 2, y: I - D / 2 },
      { x: $ + f.width / 2, y: I - D / 2 },
      { x: $ + f.width / 2, y: I - D / 2 + f.height + x },
      { x: $ + c / 2, y: I - D / 2 + f.height + x },
      { x: $ + c / 2, y: I + D / 2 },
      { x: $ - c / 2, y: I + D / 2 },
      { x: $ - c / 2, y: I - D / 2 + f.height + x },
      { x: $ - f.width / 2, y: I - D / 2 + f.height + x }
    ] : B = [
      { x: $ - c / 2, y: I - D / 2 },
      { x: $ + c / 2, y: I - D / 2 },
      { x: $ + c / 2, y: I - D / 2 + h },
      { x: $ + f.width / 2, y: I - D / 2 + h },
      { x: $ + f.width / 2 / 2, y: I + D / 2 },
      { x: $ - f.width / 2, y: I + D / 2 },
      { x: $ - f.width / 2, y: I - D / 2 + h },
      { x: $ - c / 2, y: I - D / 2 + h }
    ], q.polygon(t, B, O);
  }, u;
}
p(gd, "imageSquare");
async function md(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), s = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), l = [
    { x: 0, y: 0 },
    { x: o, y: 0 },
    { x: o + 3 * s / 6, y: -s },
    { x: -3 * s / 6, y: -s }
  ];
  let c;
  const { cssStyles: h } = t;
  if (t.look === "handDrawn") {
    const u = W.svg(a), f = H(t, {}), d = rt(l), g = u.path(d, f);
    c = a.insert(() => g, ":first-child").attr("transform", `translate(${-o / 2}, ${s / 2})`), h && c.attr("style", h);
  } else
    c = ye(a, o, s, l);
  return i && c.attr("style", i), t.width = o, t.height = s, j(t, c), t.intersect = function(u) {
    return q.polygon(t, l, u);
  }, a;
}
p(md, "inv_trapezoid");
async function bi(e, t, r) {
  const { labelStyles: i, nodeStyles: a } = Y(t);
  t.labelStyle = i;
  const { shapeSvg: n, bbox: o } = await Q(e, t, Z(t)), s = Math.max(o.width + r.labelPaddingX * 2, (t == null ? void 0 : t.width) || 0), l = Math.max(o.height + r.labelPaddingY * 2, (t == null ? void 0 : t.height) || 0), c = -s / 2, h = -l / 2;
  let u, { rx: f, ry: d } = t;
  const { cssStyles: g } = t;
  if (r != null && r.rx && r.ry && (f = r.rx, d = r.ry), t.look === "handDrawn") {
    const m = W.svg(n), y = H(t, {}), x = f || d ? m.path(me(c, h, s, l, f || 0), y) : m.rectangle(c, h, s, l, y);
    u = n.insert(() => x, ":first-child"), u.attr("class", "basic label-container").attr("style", Et(g));
  } else
    u = n.insert("rect", ":first-child"), u.attr("class", "basic label-container").attr("style", a).attr("rx", Et(f)).attr("ry", Et(d)).attr("x", c).attr("y", h).attr("width", s).attr("height", l);
  return j(t, u), t.intersect = function(m) {
    return q.rect(t, m);
  }, n;
}
p(bi, "drawRect");
async function yd(e, t) {
  const { shapeSvg: r, bbox: i, label: a } = await Q(e, t, "label"), n = r.insert("rect", ":first-child");
  return n.attr("width", 0.1).attr("height", 0.1), r.attr("class", "label edgeLabel"), a.attr(
    "transform",
    `translate(${-(i.width / 2) - (i.x - (i.left ?? 0))}, ${-(i.height / 2) - (i.y - (i.top ?? 0))})`
  ), j(t, n), t.intersect = function(l) {
    return q.rect(t, l);
  }, r;
}
p(yd, "labelRect");
async function xd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = Math.max(n.width + (t.padding ?? 0), (t == null ? void 0 : t.width) ?? 0), s = Math.max(n.height + (t.padding ?? 0), (t == null ? void 0 : t.height) ?? 0), l = [
    { x: 0, y: 0 },
    { x: o + 3 * s / 6, y: 0 },
    { x: o, y: -s },
    { x: -(3 * s) / 6, y: -s }
  ];
  let c;
  const { cssStyles: h } = t;
  if (t.look === "handDrawn") {
    const u = W.svg(a), f = H(t, {}), d = rt(l), g = u.path(d, f);
    c = a.insert(() => g, ":first-child").attr("transform", `translate(${-o / 2}, ${s / 2})`), h && c.attr("style", h);
  } else
    c = ye(a, o, s, l);
  return i && c.attr("style", i), t.width = o, t.height = s, j(t, c), t.intersect = function(u) {
    return q.polygon(t, l, u);
  }, a;
}
p(xd, "lean_left");
async function bd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = Math.max(n.width + (t.padding ?? 0), (t == null ? void 0 : t.width) ?? 0), s = Math.max(n.height + (t.padding ?? 0), (t == null ? void 0 : t.height) ?? 0), l = [
    { x: -3 * s / 6, y: 0 },
    { x: o, y: 0 },
    { x: o + 3 * s / 6, y: -s },
    { x: 0, y: -s }
  ];
  let c;
  const { cssStyles: h } = t;
  if (t.look === "handDrawn") {
    const u = W.svg(a), f = H(t, {}), d = rt(l), g = u.path(d, f);
    c = a.insert(() => g, ":first-child").attr("transform", `translate(${-o / 2}, ${s / 2})`), h && c.attr("style", h);
  } else
    c = ye(a, o, s, l);
  return i && c.attr("style", i), t.width = o, t.height = s, j(t, c), t.intersect = function(u) {
    return q.polygon(t, l, u);
  }, a;
}
p(bd, "lean_right");
function Cd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.label = "", t.labelStyle = r;
  const a = e.insert("g").attr("class", Z(t)).attr("id", t.domId ?? t.id), { cssStyles: n } = t, o = Math.max(35, (t == null ? void 0 : t.width) ?? 0), s = Math.max(35, (t == null ? void 0 : t.height) ?? 0), l = 7, c = [
    { x: o, y: 0 },
    { x: 0, y: s + l / 2 },
    { x: o - 2 * l, y: s + l / 2 },
    { x: 0, y: 2 * s },
    { x: o, y: s - l / 2 },
    { x: 2 * l, y: s - l / 2 }
  ], h = W.svg(a), u = H(t, {});
  t.look !== "handDrawn" && (u.roughness = 0, u.fillStyle = "solid");
  const f = rt(c), d = h.path(f, u), g = a.insert(() => d, ":first-child");
  return n && t.look !== "handDrawn" && g.selectAll("path").attr("style", n), i && t.look !== "handDrawn" && g.selectAll("path").attr("style", i), g.attr("transform", `translate(-${o / 2},${-s})`), j(t, g), t.intersect = function(m) {
    return F.info("lightningBolt intersect", t, m), q.polygon(t, c, m);
  }, a;
}
p(Cd, "lightningBolt");
var _w = /* @__PURE__ */ p((e, t, r, i, a, n, o) => [
  `M${e},${t + n}`,
  `a${a},${n} 0,0,0 ${r},0`,
  `a${a},${n} 0,0,0 ${-r},0`,
  `l0,${i}`,
  `a${a},${n} 0,0,0 ${r},0`,
  `l0,${-i}`,
  `M${e},${t + n + o}`,
  `a${a},${n} 0,0,0 ${r},0`
].join(" "), "createCylinderPathD"), vw = /* @__PURE__ */ p((e, t, r, i, a, n, o) => [
  `M${e},${t + n}`,
  `M${e + r},${t + n}`,
  `a${a},${n} 0,0,0 ${-r},0`,
  `l0,${i}`,
  `a${a},${n} 0,0,0 ${r},0`,
  `l0,${-i}`,
  `M${e},${t + n + o}`,
  `a${a},${n} 0,0,0 ${r},0`
].join(" "), "createOuterCylinderPathD"), Sw = /* @__PURE__ */ p((e, t, r, i, a, n) => [`M${e - r / 2},${-i / 2}`, `a${a},${n} 0,0,0 ${r},0`].join(" "), "createInnerCylinderPathD");
async function kd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0), t.width ?? 0), l = s / 2, c = l / (2.5 + s / 50), h = Math.max(n.height + c + (t.padding ?? 0), t.height ?? 0), u = h * 0.1;
  let f;
  const { cssStyles: d } = t;
  if (t.look === "handDrawn") {
    const g = W.svg(a), m = vw(0, 0, s, h, l, c, u), y = Sw(0, c, s, h, l, c), x = H(t, {}), b = g.path(m, x), k = g.path(y, x);
    a.insert(() => k, ":first-child").attr("class", "line"), f = a.insert(() => b, ":first-child"), f.attr("class", "basic label-container"), d && f.attr("style", d);
  } else {
    const g = _w(0, 0, s, h, l, c, u);
    f = a.insert("path", ":first-child").attr("d", g).attr("class", "basic label-container").attr("style", Et(d)).attr("style", i);
  }
  return f.attr("label-offset-y", c), f.attr("transform", `translate(${-s / 2}, ${-(h / 2 + c)})`), j(t, f), o.attr(
    "transform",
    `translate(${-(n.width / 2) - (n.x - (n.left ?? 0))}, ${-(n.height / 2) + c - (n.y - (n.top ?? 0))})`
  ), t.intersect = function(g) {
    const m = q.rect(t, g), y = m.x - (t.x ?? 0);
    if (l != 0 && (Math.abs(y) < (t.width ?? 0) / 2 || Math.abs(y) == (t.width ?? 0) / 2 && Math.abs(m.y - (t.y ?? 0)) > (t.height ?? 0) / 2 - c)) {
      let x = c * c * (1 - y * y / (l * l));
      x > 0 && (x = Math.sqrt(x)), x = c - x, g.y - (t.y ?? 0) > 0 && (x = -x), m.y += x;
    }
    return m;
  }, a;
}
p(kd, "linedCylinder");
async function wd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), l = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), c = l / 4, h = l + c, { cssStyles: u } = t, f = W.svg(a), d = H(t, {});
  t.look !== "handDrawn" && (d.roughness = 0, d.fillStyle = "solid");
  const g = [
    { x: -s / 2 - s / 2 * 0.1, y: -h / 2 },
    { x: -s / 2 - s / 2 * 0.1, y: h / 2 },
    ...Te(
      -s / 2 - s / 2 * 0.1,
      h / 2,
      s / 2 + s / 2 * 0.1,
      h / 2,
      c,
      0.8
    ),
    { x: s / 2 + s / 2 * 0.1, y: -h / 2 },
    { x: -s / 2 - s / 2 * 0.1, y: -h / 2 },
    { x: -s / 2, y: -h / 2 },
    { x: -s / 2, y: h / 2 * 1.1 },
    { x: -s / 2, y: -h / 2 }
  ], m = f.polygon(
    g.map((x) => [x.x, x.y]),
    d
  ), y = a.insert(() => m, ":first-child");
  return y.attr("class", "basic label-container"), u && t.look !== "handDrawn" && y.selectAll("path").attr("style", u), i && t.look !== "handDrawn" && y.selectAll("path").attr("style", i), y.attr("transform", `translate(0,${-c / 2})`), o.attr(
    "transform",
    `translate(${-s / 2 + (t.padding ?? 0) + s / 2 * 0.1 / 2 - (n.x - (n.left ?? 0))},${-l / 2 + (t.padding ?? 0) - c / 2 - (n.y - (n.top ?? 0))})`
  ), j(t, y), t.intersect = function(x) {
    return q.polygon(t, g, x);
  }, a;
}
p(wd, "linedWaveEdgedRect");
async function _d(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), l = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), c = 5, h = -s / 2, u = -l / 2, { cssStyles: f } = t, d = W.svg(a), g = H(t, {}), m = [
    { x: h - c, y: u + c },
    { x: h - c, y: u + l + c },
    { x: h + s - c, y: u + l + c },
    { x: h + s - c, y: u + l },
    { x: h + s, y: u + l },
    { x: h + s, y: u + l - c },
    { x: h + s + c, y: u + l - c },
    { x: h + s + c, y: u - c },
    { x: h + c, y: u - c },
    { x: h + c, y: u },
    { x: h, y: u },
    { x: h, y: u + c }
  ], y = [
    { x: h, y: u + c },
    { x: h + s - c, y: u + c },
    { x: h + s - c, y: u + l },
    { x: h + s, y: u + l },
    { x: h + s, y: u },
    { x: h, y: u }
  ];
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const x = rt(m), b = d.path(x, g), k = rt(y), S = d.path(k, { ...g, fill: "none" }), w = a.insert(() => S, ":first-child");
  return w.insert(() => b, ":first-child"), w.attr("class", "basic label-container"), f && t.look !== "handDrawn" && w.selectAll("path").attr("style", f), i && t.look !== "handDrawn" && w.selectAll("path").attr("style", i), o.attr(
    "transform",
    `translate(${-(n.width / 2) - c - (n.x - (n.left ?? 0))}, ${-(n.height / 2) + c - (n.y - (n.top ?? 0))})`
  ), j(t, w), t.intersect = function(C) {
    return q.polygon(t, m, C);
  }, a;
}
p(_d, "multiRect");
async function vd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), l = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), c = l / 4, h = l + c, u = -s / 2, f = -h / 2, d = 5, { cssStyles: g } = t, m = Te(
    u - d,
    f + h + d,
    u + s - d,
    f + h + d,
    c,
    0.8
  ), y = m == null ? void 0 : m[m.length - 1], x = [
    { x: u - d, y: f + d },
    { x: u - d, y: f + h + d },
    ...m,
    { x: u + s - d, y: y.y - d },
    { x: u + s, y: y.y - d },
    { x: u + s, y: y.y - 2 * d },
    { x: u + s + d, y: y.y - 2 * d },
    { x: u + s + d, y: f - d },
    { x: u + d, y: f - d },
    { x: u + d, y: f },
    { x: u, y: f },
    { x: u, y: f + d }
  ], b = [
    { x: u, y: f + d },
    { x: u + s - d, y: f + d },
    { x: u + s - d, y: y.y - d },
    { x: u + s, y: y.y - d },
    { x: u + s, y: f },
    { x: u, y: f }
  ], k = W.svg(a), S = H(t, {});
  t.look !== "handDrawn" && (S.roughness = 0, S.fillStyle = "solid");
  const w = rt(x), C = k.path(w, S), _ = rt(b), E = k.path(_, S), R = a.insert(() => C, ":first-child");
  return R.insert(() => E), R.attr("class", "basic label-container"), g && t.look !== "handDrawn" && R.selectAll("path").attr("style", g), i && t.look !== "handDrawn" && R.selectAll("path").attr("style", i), R.attr("transform", `translate(0,${-c / 2})`), o.attr(
    "transform",
    `translate(${-(n.width / 2) - d - (n.x - (n.left ?? 0))}, ${-(n.height / 2) + d - c / 2 - (n.y - (n.top ?? 0))})`
  ), j(t, R), t.intersect = function(O) {
    return q.polygon(t, x, O);
  }, a;
}
p(vd, "multiWaveEdgedRectangle");
async function Sd(e, t, { config: { themeVariables: r } }) {
  var b;
  const { labelStyles: i, nodeStyles: a } = Y(t);
  t.labelStyle = i, t.useHtmlLabels || ((b = It().flowchart) == null ? void 0 : b.htmlLabels) !== !1 || (t.centerLabel = !0);
  const { shapeSvg: o, bbox: s, label: l } = await Q(e, t, Z(t)), c = Math.max(s.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), h = Math.max(s.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), u = -c / 2, f = -h / 2, { cssStyles: d } = t, g = W.svg(o), m = H(t, {
    fill: r.noteBkgColor,
    stroke: r.noteBorderColor
  });
  t.look !== "handDrawn" && (m.roughness = 0, m.fillStyle = "solid");
  const y = g.rectangle(u, f, c, h, m), x = o.insert(() => y, ":first-child");
  return x.attr("class", "basic label-container"), d && t.look !== "handDrawn" && x.selectAll("path").attr("style", d), a && t.look !== "handDrawn" && x.selectAll("path").attr("style", a), l.attr(
    "transform",
    `translate(${-s.width / 2 - (s.x - (s.left ?? 0))}, ${-(s.height / 2) - (s.y - (s.top ?? 0))})`
  ), j(t, x), t.intersect = function(k) {
    return q.rect(t, k);
  }, o;
}
p(Sd, "note");
var Tw = /* @__PURE__ */ p((e, t, r) => [
  `M${e + r / 2},${t}`,
  `L${e + r},${t - r / 2}`,
  `L${e + r / 2},${t - r}`,
  `L${e},${t - r / 2}`,
  "Z"
].join(" "), "createDecisionBoxPathD");
async function Td(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = n.width + t.padding, s = n.height + t.padding, l = o + s, c = [
    { x: l / 2, y: 0 },
    { x: l, y: -l / 2 },
    { x: l / 2, y: -l },
    { x: 0, y: -l / 2 }
  ];
  let h;
  const { cssStyles: u } = t;
  if (t.look === "handDrawn") {
    const f = W.svg(a), d = H(t, {}), g = Tw(0, 0, l), m = f.path(g, d);
    h = a.insert(() => m, ":first-child").attr("transform", `translate(${-l / 2}, ${l / 2})`), u && h.attr("style", u);
  } else
    h = ye(a, l, l, c);
  return i && h.attr("style", i), j(t, h), t.intersect = function(f) {
    return F.debug(
      `APA12 Intersect called SPLIT
point:`,
      f,
      `
node:
`,
      t,
      `
res:`,
      q.polygon(t, c, f)
    ), q.polygon(t, c, f);
  }, a;
}
p(Td, "question");
async function Bd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0), (t == null ? void 0 : t.width) ?? 0), l = Math.max(n.height + (t.padding ?? 0), (t == null ? void 0 : t.height) ?? 0), c = -s / 2, h = -l / 2, u = h / 2, f = [
    { x: c + u, y: h },
    { x: c, y: 0 },
    { x: c + u, y: -h },
    { x: -c, y: -h },
    { x: -c, y: h }
  ], { cssStyles: d } = t, g = W.svg(a), m = H(t, {});
  t.look !== "handDrawn" && (m.roughness = 0, m.fillStyle = "solid");
  const y = rt(f), x = g.path(y, m), b = a.insert(() => x, ":first-child");
  return b.attr("class", "basic label-container"), d && t.look !== "handDrawn" && b.selectAll("path").attr("style", d), i && t.look !== "handDrawn" && b.selectAll("path").attr("style", i), b.attr("transform", `translate(${-u / 2},0)`), o.attr(
    "transform",
    `translate(${-u / 2 - n.width / 2 - (n.x - (n.left ?? 0))}, ${-(n.height / 2) - (n.y - (n.top ?? 0))})`
  ), j(t, b), t.intersect = function(k) {
    return q.polygon(t, f, k);
  }, a;
}
p(Bd, "rect_left_inv_arrow");
async function Ld(e, t) {
  var E, R;
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  let a;
  t.cssClasses ? a = "node " + t.cssClasses : a = "node default";
  const n = e.insert("g").attr("class", a).attr("id", t.domId || t.id), o = n.insert("g"), s = n.insert("g").attr("class", "label").attr("style", i), l = t.description, c = t.label, h = s.node().appendChild(await Pe(c, t.labelStyle, !0, !0));
  let u = { width: 0, height: 0 };
  if (bt((R = (E = at()) == null ? void 0 : E.flowchart) == null ? void 0 : R.htmlLabels)) {
    const O = h.children[0], $ = et(h);
    u = O.getBoundingClientRect(), $.attr("width", u.width), $.attr("height", u.height);
  }
  F.info("Text 2", l);
  const f = l || [], d = h.getBBox(), g = s.node().appendChild(
    await Pe(
      f.join ? f.join("<br/>") : f,
      t.labelStyle,
      !0,
      !0
    )
  ), m = g.children[0], y = et(g);
  u = m.getBoundingClientRect(), y.attr("width", u.width), y.attr("height", u.height);
  const x = (t.padding || 0) / 2;
  et(g).attr(
    "transform",
    "translate( " + (u.width > d.width ? 0 : (d.width - u.width) / 2) + ", " + (d.height + x + 5) + ")"
  ), et(h).attr(
    "transform",
    "translate( " + (u.width < d.width ? 0 : -(d.width - u.width) / 2) + ", 0)"
  ), u = s.node().getBBox(), s.attr(
    "transform",
    "translate(" + -u.width / 2 + ", " + (-u.height / 2 - x + 3) + ")"
  );
  const b = u.width + (t.padding || 0), k = u.height + (t.padding || 0), S = -u.width / 2 - x, w = -u.height / 2 - x;
  let C, _;
  if (t.look === "handDrawn") {
    const O = W.svg(n), $ = H(t, {}), I = O.path(
      me(S, w, b, k, t.rx || 0),
      $
    ), D = O.line(
      -u.width / 2 - x,
      -u.height / 2 - x + d.height + x,
      u.width / 2 + x,
      -u.height / 2 - x + d.height + x,
      $
    );
    _ = n.insert(() => (F.debug("Rough node insert CXC", I), D), ":first-child"), C = n.insert(() => (F.debug("Rough node insert CXC", I), I), ":first-child");
  } else
    C = o.insert("rect", ":first-child"), _ = o.insert("line"), C.attr("class", "outer title-state").attr("style", i).attr("x", -u.width / 2 - x).attr("y", -u.height / 2 - x).attr("width", u.width + (t.padding || 0)).attr("height", u.height + (t.padding || 0)), _.attr("class", "divider").attr("x1", -u.width / 2 - x).attr("x2", u.width / 2 + x).attr("y1", -u.height / 2 - x + d.height + x).attr("y2", -u.height / 2 - x + d.height + x);
  return j(t, C), t.intersect = function(O) {
    return q.rect(t, O);
  }, n;
}
p(Ld, "rectWithTitle");
async function Md(e, t) {
  const r = {
    rx: 5,
    ry: 5,
    labelPaddingX: ((t == null ? void 0 : t.padding) || 0) * 1,
    labelPaddingY: ((t == null ? void 0 : t.padding) || 0) * 1
  };
  return bi(e, t, r);
}
p(Md, "roundedRect");
async function $d(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = (t == null ? void 0 : t.padding) ?? 0, l = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), c = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), h = -n.width / 2 - s, u = -n.height / 2 - s, { cssStyles: f } = t, d = W.svg(a), g = H(t, {});
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const m = [
    { x: h, y: u },
    { x: h + l + 8, y: u },
    { x: h + l + 8, y: u + c },
    { x: h - 8, y: u + c },
    { x: h - 8, y: u },
    { x: h, y: u },
    { x: h, y: u + c }
  ], y = d.polygon(
    m.map((b) => [b.x, b.y]),
    g
  ), x = a.insert(() => y, ":first-child");
  return x.attr("class", "basic label-container").attr("style", Et(f)), i && t.look !== "handDrawn" && x.selectAll("path").attr("style", i), f && t.look !== "handDrawn" && x.selectAll("path").attr("style", i), o.attr(
    "transform",
    `translate(${-l / 2 + 4 + (t.padding ?? 0) - (n.x - (n.left ?? 0))},${-c / 2 + (t.padding ?? 0) - (n.y - (n.top ?? 0))})`
  ), j(t, x), t.intersect = function(b) {
    return q.rect(t, b);
  }, a;
}
p($d, "shadedProcess");
async function Ad(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), l = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), c = -s / 2, h = -l / 2, { cssStyles: u } = t, f = W.svg(a), d = H(t, {});
  t.look !== "handDrawn" && (d.roughness = 0, d.fillStyle = "solid");
  const g = [
    { x: c, y: h },
    { x: c, y: h + l },
    { x: c + s, y: h + l },
    { x: c + s, y: h - l / 2 }
  ], m = rt(g), y = f.path(m, d), x = a.insert(() => y, ":first-child");
  return x.attr("class", "basic label-container"), u && t.look !== "handDrawn" && x.selectChildren("path").attr("style", u), i && t.look !== "handDrawn" && x.selectChildren("path").attr("style", i), x.attr("transform", `translate(0, ${l / 4})`), o.attr(
    "transform",
    `translate(${-s / 2 + (t.padding ?? 0) - (n.x - (n.left ?? 0))}, ${-l / 4 + (t.padding ?? 0) - (n.y - (n.top ?? 0))})`
  ), j(t, x), t.intersect = function(b) {
    return q.polygon(t, g, b);
  }, a;
}
p(Ad, "slopedRect");
async function Fd(e, t) {
  const r = {
    rx: 0,
    ry: 0,
    labelPaddingX: ((t == null ? void 0 : t.padding) || 0) * 2,
    labelPaddingY: ((t == null ? void 0 : t.padding) || 0) * 1
  };
  return bi(e, t, r);
}
p(Fd, "squareRect");
async function Ed(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = n.height + t.padding, s = n.width + o / 4 + t.padding;
  let l;
  const { cssStyles: c } = t;
  if (t.look === "handDrawn") {
    const h = W.svg(a), u = H(t, {}), f = me(-s / 2, -o / 2, s, o, o / 2), d = h.path(f, u);
    l = a.insert(() => d, ":first-child"), l.attr("class", "basic label-container").attr("style", Et(c));
  } else
    l = a.insert("rect", ":first-child"), l.attr("class", "basic label-container").attr("style", i).attr("rx", o / 2).attr("ry", o / 2).attr("x", -s / 2).attr("y", -o / 2).attr("width", s).attr("height", o);
  return j(t, l), t.intersect = function(h) {
    return q.rect(t, h);
  }, a;
}
p(Ed, "stadium");
async function Od(e, t) {
  return bi(e, t, {
    rx: 5,
    ry: 5
  });
}
p(Od, "state");
function Dd(e, t, { config: { themeVariables: r } }) {
  const { labelStyles: i, nodeStyles: a } = Y(t);
  t.labelStyle = i;
  const { cssStyles: n } = t, { lineColor: o, stateBorder: s, nodeBorder: l } = r, c = e.insert("g").attr("class", "node default").attr("id", t.domId || t.id), h = W.svg(c), u = H(t, {});
  t.look !== "handDrawn" && (u.roughness = 0, u.fillStyle = "solid");
  const f = h.circle(0, 0, 14, {
    ...u,
    stroke: o,
    strokeWidth: 2
  }), d = s ?? l, g = h.circle(0, 0, 5, {
    ...u,
    fill: d,
    stroke: d,
    strokeWidth: 2,
    fillStyle: "solid"
  }), m = c.insert(() => f, ":first-child");
  return m.insert(() => g), n && m.selectAll("path").attr("style", n), a && m.selectAll("path").attr("style", a), j(t, m), t.intersect = function(y) {
    return q.circle(t, 7, y);
  }, c;
}
p(Dd, "stateEnd");
function Rd(e, t, { config: { themeVariables: r } }) {
  const { lineColor: i } = r, a = e.insert("g").attr("class", "node default").attr("id", t.domId || t.id);
  let n;
  if (t.look === "handDrawn") {
    const s = W.svg(a).circle(0, 0, 14, hm(i));
    n = a.insert(() => s), n.attr("class", "state-start").attr("r", 7).attr("width", 14).attr("height", 14);
  } else
    n = a.insert("circle", ":first-child"), n.attr("class", "state-start").attr("r", 7).attr("width", 14).attr("height", 14);
  return j(t, n), t.intersect = function(o) {
    return q.circle(t, 7, o);
  }, a;
}
p(Rd, "stateStart");
async function Pd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = ((t == null ? void 0 : t.padding) || 0) / 2, s = n.width + t.padding, l = n.height + t.padding, c = -n.width / 2 - o, h = -n.height / 2 - o, u = [
    { x: 0, y: 0 },
    { x: s, y: 0 },
    { x: s, y: -l },
    { x: 0, y: -l },
    { x: 0, y: 0 },
    { x: -8, y: 0 },
    { x: s + 8, y: 0 },
    { x: s + 8, y: -l },
    { x: -8, y: -l },
    { x: -8, y: 0 }
  ];
  if (t.look === "handDrawn") {
    const f = W.svg(a), d = H(t, {}), g = f.rectangle(c - 8, h, s + 16, l, d), m = f.line(c, h, c, h + l, d), y = f.line(c + s, h, c + s, h + l, d);
    a.insert(() => m, ":first-child"), a.insert(() => y, ":first-child");
    const x = a.insert(() => g, ":first-child"), { cssStyles: b } = t;
    x.attr("class", "basic label-container").attr("style", Et(b)), j(t, x);
  } else {
    const f = ye(a, s, l, u);
    i && f.attr("style", i), j(t, f);
  }
  return t.intersect = function(f) {
    return q.polygon(t, u, f);
  }, a;
}
p(Pd, "subroutine");
async function Id(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), s = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), l = -o / 2, c = -s / 2, h = 0.2 * s, u = 0.2 * s, { cssStyles: f } = t, d = W.svg(a), g = H(t, {}), m = [
    { x: l - h / 2, y: c },
    { x: l + o + h / 2, y: c },
    { x: l + o + h / 2, y: c + s },
    { x: l - h / 2, y: c + s }
  ], y = [
    { x: l + o - h / 2, y: c + s },
    { x: l + o + h / 2, y: c + s },
    { x: l + o + h / 2, y: c + s - u }
  ];
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const x = rt(m), b = d.path(x, g), k = rt(y), S = d.path(k, { ...g, fillStyle: "solid" }), w = a.insert(() => S, ":first-child");
  return w.insert(() => b, ":first-child"), w.attr("class", "basic label-container"), f && t.look !== "handDrawn" && w.selectAll("path").attr("style", f), i && t.look !== "handDrawn" && w.selectAll("path").attr("style", i), j(t, w), t.intersect = function(C) {
    return q.polygon(t, m, C);
  }, a;
}
p(Id, "taggedRect");
async function Nd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), l = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), c = l / 4, h = 0.2 * s, u = 0.2 * l, f = l + c, { cssStyles: d } = t, g = W.svg(a), m = H(t, {});
  t.look !== "handDrawn" && (m.roughness = 0, m.fillStyle = "solid");
  const y = [
    { x: -s / 2 - s / 2 * 0.1, y: f / 2 },
    ...Te(
      -s / 2 - s / 2 * 0.1,
      f / 2,
      s / 2 + s / 2 * 0.1,
      f / 2,
      c,
      0.8
    ),
    { x: s / 2 + s / 2 * 0.1, y: -f / 2 },
    { x: -s / 2 - s / 2 * 0.1, y: -f / 2 }
  ], x = -s / 2 + s / 2 * 0.1, b = -f / 2 - u * 0.4, k = [
    { x: x + s - h, y: (b + l) * 1.4 },
    { x: x + s, y: b + l - u },
    { x: x + s, y: (b + l) * 0.9 },
    ...Te(
      x + s,
      (b + l) * 1.3,
      x + s - h,
      (b + l) * 1.5,
      -l * 0.03,
      0.5
    )
  ], S = rt(y), w = g.path(S, m), C = rt(k), _ = g.path(C, {
    ...m,
    fillStyle: "solid"
  }), E = a.insert(() => _, ":first-child");
  return E.insert(() => w, ":first-child"), E.attr("class", "basic label-container"), d && t.look !== "handDrawn" && E.selectAll("path").attr("style", d), i && t.look !== "handDrawn" && E.selectAll("path").attr("style", i), E.attr("transform", `translate(0,${-c / 2})`), o.attr(
    "transform",
    `translate(${-s / 2 + (t.padding ?? 0) - (n.x - (n.left ?? 0))},${-l / 2 + (t.padding ?? 0) - c / 2 - (n.y - (n.top ?? 0))})`
  ), j(t, E), t.intersect = function(R) {
    return q.polygon(t, y, R);
  }, a;
}
p(Nd, "taggedWaveEdgedRectangle");
async function zd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = Math.max(n.width + t.padding, (t == null ? void 0 : t.width) || 0), s = Math.max(n.height + t.padding, (t == null ? void 0 : t.height) || 0), l = -o / 2, c = -s / 2, h = a.insert("rect", ":first-child");
  return h.attr("class", "text").attr("style", i).attr("rx", 0).attr("ry", 0).attr("x", l).attr("y", c).attr("width", o).attr("height", s), j(t, h), t.intersect = function(u) {
    return q.rect(t, u);
  }, a;
}
p(zd, "text");
var Bw = /* @__PURE__ */ p((e, t, r, i, a, n) => `M${e},${t}
    a${a},${n} 0,0,1 0,${-i}
    l${r},0
    a${a},${n} 0,0,1 0,${i}
    M${r},${-i}
    a${a},${n} 0,0,0 0,${i}
    l${-r},0`, "createCylinderPathD"), Lw = /* @__PURE__ */ p((e, t, r, i, a, n) => [
  `M${e},${t}`,
  `M${e + r},${t}`,
  `a${a},${n} 0,0,0 0,${-i}`,
  `l${-r},0`,
  `a${a},${n} 0,0,0 0,${i}`,
  `l${r},0`
].join(" "), "createOuterCylinderPathD"), Mw = /* @__PURE__ */ p((e, t, r, i, a, n) => [`M${e + r / 2},${-i / 2}`, `a${a},${n} 0,0,0 0,${i}`].join(" "), "createInnerCylinderPathD");
async function qd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o, halfPadding: s } = await Q(
    e,
    t,
    Z(t)
  ), l = t.look === "neo" ? s * 2 : s, c = n.height + l, h = c / 2, u = h / (2.5 + c / 50), f = n.width + u + l, { cssStyles: d } = t;
  let g;
  if (t.look === "handDrawn") {
    const m = W.svg(a), y = Lw(0, 0, f, c, u, h), x = Mw(0, 0, f, c, u, h), b = m.path(y, H(t, {})), k = m.path(x, H(t, { fill: "none" }));
    g = a.insert(() => k, ":first-child"), g = a.insert(() => b, ":first-child"), g.attr("class", "basic label-container"), d && g.attr("style", d);
  } else {
    const m = Bw(0, 0, f, c, u, h);
    g = a.insert("path", ":first-child").attr("d", m).attr("class", "basic label-container").attr("style", Et(d)).attr("style", i), g.attr("class", "basic label-container"), d && g.selectAll("path").attr("style", d), i && g.selectAll("path").attr("style", i);
  }
  return g.attr("label-offset-x", u), g.attr("transform", `translate(${-f / 2}, ${c / 2} )`), o.attr(
    "transform",
    `translate(${-(n.width / 2) - u - (n.x - (n.left ?? 0))}, ${-(n.height / 2) - (n.y - (n.top ?? 0))})`
  ), j(t, g), t.intersect = function(m) {
    const y = q.rect(t, m), x = y.y - (t.y ?? 0);
    if (h != 0 && (Math.abs(x) < (t.height ?? 0) / 2 || Math.abs(x) == (t.height ?? 0) / 2 && Math.abs(y.x - (t.x ?? 0)) > (t.width ?? 0) / 2 - u)) {
      let b = u * u * (1 - x * x / (h * h));
      b != 0 && (b = Math.sqrt(Math.abs(b))), b = u - b, m.x - (t.x ?? 0) > 0 && (b = -b), y.x += b;
    }
    return y;
  }, a;
}
p(qd, "tiltedCylinder");
async function Wd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = n.width + t.padding, s = n.height + t.padding, l = [
    { x: -3 * s / 6, y: 0 },
    { x: o + 3 * s / 6, y: 0 },
    { x: o, y: -s },
    { x: 0, y: -s }
  ];
  let c;
  const { cssStyles: h } = t;
  if (t.look === "handDrawn") {
    const u = W.svg(a), f = H(t, {}), d = rt(l), g = u.path(d, f);
    c = a.insert(() => g, ":first-child").attr("transform", `translate(${-o / 2}, ${s / 2})`), h && c.attr("style", h);
  } else
    c = ye(a, o, s, l);
  return i && c.attr("style", i), t.width = o, t.height = s, j(t, c), t.intersect = function(u) {
    return q.polygon(t, l, u);
  }, a;
}
p(Wd, "trapezoid");
async function Hd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = 60, s = 20, l = Math.max(o, n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), c = Math.max(s, n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), { cssStyles: h } = t, u = W.svg(a), f = H(t, {});
  t.look !== "handDrawn" && (f.roughness = 0, f.fillStyle = "solid");
  const d = [
    { x: -l / 2 * 0.8, y: -c / 2 },
    { x: l / 2 * 0.8, y: -c / 2 },
    { x: l / 2, y: -c / 2 * 0.6 },
    { x: l / 2, y: c / 2 },
    { x: -l / 2, y: c / 2 },
    { x: -l / 2, y: -c / 2 * 0.6 }
  ], g = rt(d), m = u.path(g, f), y = a.insert(() => m, ":first-child");
  return y.attr("class", "basic label-container"), h && t.look !== "handDrawn" && y.selectChildren("path").attr("style", h), i && t.look !== "handDrawn" && y.selectChildren("path").attr("style", i), j(t, y), t.intersect = function(x) {
    return q.polygon(t, d, x);
  }, a;
}
p(Hd, "trapezoidalPentagon");
async function jd(e, t) {
  var b;
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = bt((b = at().flowchart) == null ? void 0 : b.htmlLabels), l = n.width + (t.padding ?? 0), c = l + n.height, h = l + n.height, u = [
    { x: 0, y: 0 },
    { x: h, y: 0 },
    { x: h / 2, y: -c }
  ], { cssStyles: f } = t, d = W.svg(a), g = H(t, {});
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const m = rt(u), y = d.path(m, g), x = a.insert(() => y, ":first-child").attr("transform", `translate(${-c / 2}, ${c / 2})`);
  return f && t.look !== "handDrawn" && x.selectChildren("path").attr("style", f), i && t.look !== "handDrawn" && x.selectChildren("path").attr("style", i), t.width = l, t.height = c, j(t, x), o.attr(
    "transform",
    `translate(${-n.width / 2 - (n.x - (n.left ?? 0))}, ${c / 2 - (n.height + (t.padding ?? 0) / (s ? 2 : 1) - (n.y - (n.top ?? 0)))})`
  ), t.intersect = function(k) {
    return F.info("Triangle intersect", t, u, k), q.polygon(t, u, k);
  }, a;
}
p(jd, "triangle");
async function Yd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), l = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), c = l / 8, h = l + c, { cssStyles: u } = t, d = 70 - s, g = d > 0 ? d / 2 : 0, m = W.svg(a), y = H(t, {});
  t.look !== "handDrawn" && (y.roughness = 0, y.fillStyle = "solid");
  const x = [
    { x: -s / 2 - g, y: h / 2 },
    ...Te(
      -s / 2 - g,
      h / 2,
      s / 2 + g,
      h / 2,
      c,
      0.8
    ),
    { x: s / 2 + g, y: -h / 2 },
    { x: -s / 2 - g, y: -h / 2 }
  ], b = rt(x), k = m.path(b, y), S = a.insert(() => k, ":first-child");
  return S.attr("class", "basic label-container"), u && t.look !== "handDrawn" && S.selectAll("path").attr("style", u), i && t.look !== "handDrawn" && S.selectAll("path").attr("style", i), S.attr("transform", `translate(0,${-c / 2})`), o.attr(
    "transform",
    `translate(${-s / 2 + (t.padding ?? 0) - (n.x - (n.left ?? 0))},${-l / 2 + (t.padding ?? 0) - c - (n.y - (n.top ?? 0))})`
  ), j(t, S), t.intersect = function(w) {
    return q.polygon(t, x, w);
  }, a;
}
p(Yd, "waveEdgedRectangle");
async function Gd(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n } = await Q(e, t, Z(t)), o = 100, s = 50, l = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), c = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), h = l / c;
  let u = l, f = c;
  u > f * h ? f = u / h : u = f * h, u = Math.max(u, o), f = Math.max(f, s);
  const d = Math.min(f * 0.2, f / 4), g = f + d * 2, { cssStyles: m } = t, y = W.svg(a), x = H(t, {});
  t.look !== "handDrawn" && (x.roughness = 0, x.fillStyle = "solid");
  const b = [
    { x: -u / 2, y: g / 2 },
    ...Te(-u / 2, g / 2, u / 2, g / 2, d, 1),
    { x: u / 2, y: -g / 2 },
    ...Te(u / 2, -g / 2, -u / 2, -g / 2, d, -1)
  ], k = rt(b), S = y.path(k, x), w = a.insert(() => S, ":first-child");
  return w.attr("class", "basic label-container"), m && t.look !== "handDrawn" && w.selectAll("path").attr("style", m), i && t.look !== "handDrawn" && w.selectAll("path").attr("style", i), j(t, w), t.intersect = function(C) {
    return q.polygon(t, b, C);
  }, a;
}
p(Gd, "waveRectangle");
async function Ud(e, t) {
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const { shapeSvg: a, bbox: n, label: o } = await Q(e, t, Z(t)), s = Math.max(n.width + (t.padding ?? 0) * 2, (t == null ? void 0 : t.width) ?? 0), l = Math.max(n.height + (t.padding ?? 0) * 2, (t == null ? void 0 : t.height) ?? 0), c = 5, h = -s / 2, u = -l / 2, { cssStyles: f } = t, d = W.svg(a), g = H(t, {}), m = [
    { x: h - c, y: u - c },
    { x: h - c, y: u + l },
    { x: h + s, y: u + l },
    { x: h + s, y: u - c }
  ], y = `M${h - c},${u - c} L${h + s},${u - c} L${h + s},${u + l} L${h - c},${u + l} L${h - c},${u - c}
                M${h - c},${u} L${h + s},${u}
                M${h},${u - c} L${h},${u + l}`;
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const x = d.path(y, g), b = a.insert(() => x, ":first-child");
  return b.attr("transform", `translate(${c / 2}, ${c / 2})`), b.attr("class", "basic label-container"), f && t.look !== "handDrawn" && b.selectAll("path").attr("style", f), i && t.look !== "handDrawn" && b.selectAll("path").attr("style", i), o.attr(
    "transform",
    `translate(${-(n.width / 2) + c / 2 - (n.x - (n.left ?? 0))}, ${-(n.height / 2) + c / 2 - (n.y - (n.top ?? 0))})`
  ), j(t, b), t.intersect = function(k) {
    return q.polygon(t, m, k);
  }, a;
}
p(Ud, "windowPane");
async function oo(e, t) {
  var lt, ft, yt, Mt;
  const r = t;
  if (r.alias && (t.label = r.alias), t.look === "handDrawn") {
    const { themeVariables: tt } = It(), { background: it } = tt, gt = {
      ...t,
      id: t.id + "-background",
      look: "default",
      cssStyles: ["stroke: none", `fill: ${it}`]
    };
    await oo(e, gt);
  }
  const i = It();
  t.useHtmlLabels = i.htmlLabels;
  let a = ((lt = i.er) == null ? void 0 : lt.diagramPadding) ?? 10, n = ((ft = i.er) == null ? void 0 : ft.entityPadding) ?? 6;
  const { cssStyles: o } = t, { labelStyles: s, nodeStyles: l } = Y(t);
  if (r.attributes.length === 0 && t.label) {
    const tt = {
      rx: 0,
      ry: 0,
      labelPaddingX: a,
      labelPaddingY: a * 1.5
    };
    pe(t.label, i) + tt.labelPaddingX * 2 < i.er.minEntityWidth && (t.width = i.er.minEntityWidth);
    const it = await bi(e, t, tt);
    if (!bt(i.htmlLabels)) {
      const gt = it.select("text"), dt = (yt = gt.node()) == null ? void 0 : yt.getBBox();
      gt.attr("transform", `translate(${-dt.width / 2}, 0)`);
    }
    return it;
  }
  i.htmlLabels || (a *= 1.25, n *= 1.25);
  let c = Z(t);
  c || (c = "node default");
  const h = e.insert("g").attr("class", c).attr("id", t.domId || t.id), u = await er(h, t.label ?? "", i, 0, 0, ["name"], s);
  u.height += n;
  let f = 0;
  const d = [], g = [];
  let m = 0, y = 0, x = 0, b = 0, k = !0, S = !0;
  for (const tt of r.attributes) {
    const it = await er(
      h,
      tt.type,
      i,
      0,
      f,
      ["attribute-type"],
      s
    );
    m = Math.max(m, it.width + a);
    const gt = await er(
      h,
      tt.name,
      i,
      0,
      f,
      ["attribute-name"],
      s
    );
    y = Math.max(y, gt.width + a);
    const dt = await er(
      h,
      tt.keys.join(),
      i,
      0,
      f,
      ["attribute-keys"],
      s
    );
    x = Math.max(x, dt.width + a);
    const ut = await er(
      h,
      tt.comment,
      i,
      0,
      f,
      ["attribute-comment"],
      s
    );
    b = Math.max(b, ut.width + a);
    const kt = Math.max(it.height, gt.height, dt.height, ut.height) + n;
    g.push({ yOffset: f, rowHeight: kt }), f += kt;
  }
  let w = 4;
  x <= a && (k = !1, x = 0, w--), b <= a && (S = !1, b = 0, w--);
  const C = h.node().getBBox();
  if (u.width + a * 2 - (m + y + x + b) > 0) {
    const tt = u.width + a * 2 - (m + y + x + b);
    m += tt / w, y += tt / w, x > 0 && (x += tt / w), b > 0 && (b += tt / w);
  }
  const _ = m + y + x + b, E = W.svg(h), R = H(t, {});
  t.look !== "handDrawn" && (R.roughness = 0, R.fillStyle = "solid");
  let O = 0;
  g.length > 0 && (O = g.reduce((tt, it) => tt + ((it == null ? void 0 : it.rowHeight) ?? 0), 0));
  const $ = Math.max(C.width + a * 2, (t == null ? void 0 : t.width) || 0, _), I = Math.max((O ?? 0) + u.height, (t == null ? void 0 : t.height) || 0), D = -$ / 2, B = -I / 2;
  h.selectAll("g:not(:first-child)").each((tt, it, gt) => {
    const dt = et(gt[it]), ut = dt.attr("transform");
    let kt = 0, xe = 0;
    if (ut) {
      const Ka = RegExp(/translate\(([^,]+),([^)]+)\)/).exec(ut);
      Ka && (kt = parseFloat(Ka[1]), xe = parseFloat(Ka[2]), dt.attr("class").includes("attribute-name") ? kt += m : dt.attr("class").includes("attribute-keys") ? kt += m + y : dt.attr("class").includes("attribute-comment") && (kt += m + y + x));
    }
    dt.attr(
      "transform",
      `translate(${D + a / 2 + kt}, ${xe + B + u.height + n / 2})`
    );
  }), h.select(".name").attr("transform", "translate(" + -u.width / 2 + ", " + (B + n / 2) + ")");
  const M = E.rectangle(D, B, $, I, R), T = h.insert(() => M, ":first-child").attr("style", o.join("")), { themeVariables: A } = It(), { rowEven: L, rowOdd: N, nodeBorder: U } = A;
  d.push(0);
  for (const [tt, it] of g.entries()) {
    const dt = (tt + 1) % 2 === 0 && it.yOffset !== 0, ut = E.rectangle(D, u.height + B + (it == null ? void 0 : it.yOffset), $, it == null ? void 0 : it.rowHeight, {
      ...R,
      fill: dt ? L : N,
      stroke: U
    });
    h.insert(() => ut, "g.label").attr("style", o.join("")).attr("class", `row-rect-${dt ? "even" : "odd"}`);
  }
  let J = E.line(D, u.height + B, $ + D, u.height + B, R);
  h.insert(() => J).attr("class", "divider"), J = E.line(m + D, u.height + B, m + D, I + B, R), h.insert(() => J).attr("class", "divider"), k && (J = E.line(
    m + y + D,
    u.height + B,
    m + y + D,
    I + B,
    R
  ), h.insert(() => J).attr("class", "divider")), S && (J = E.line(
    m + y + x + D,
    u.height + B,
    m + y + x + D,
    I + B,
    R
  ), h.insert(() => J).attr("class", "divider"));
  for (const tt of d)
    J = E.line(
      D,
      u.height + B + tt,
      $ + D,
      u.height + B + tt,
      R
    ), h.insert(() => J).attr("class", "divider");
  if (j(t, T), l && t.look !== "handDrawn") {
    const tt = l.split(";"), it = (Mt = tt == null ? void 0 : tt.filter((gt) => gt.includes("stroke"))) == null ? void 0 : Mt.map((gt) => `${gt}`).join("; ");
    h.selectAll("path").attr("style", it ?? ""), h.selectAll(".row-rect-even path").attr("style", l);
  }
  return t.intersect = function(tt) {
    return q.rect(t, tt);
  }, h;
}
p(oo, "erBox");
async function er(e, t, r, i = 0, a = 0, n = [], o = "") {
  const s = e.insert("g").attr("class", `label ${n.join(" ")}`).attr("transform", `translate(${i}, ${a})`).attr("style", o);
  t !== go(t) && (t = go(t), t = t.replaceAll("<", "&lt;").replaceAll(">", "&gt;"));
  const l = s.node().appendChild(
    await Le(
      s,
      t,
      {
        width: pe(t, r) + 100,
        style: o,
        useHtmlLabels: r.htmlLabels
      },
      r
    )
  );
  if (t.includes("&lt;") || t.includes("&gt;")) {
    let h = l.children[0];
    for (h.textContent = h.textContent.replaceAll("&lt;", "<").replaceAll("&gt;", ">"); h.childNodes[0]; )
      h = h.childNodes[0], h.textContent = h.textContent.replaceAll("&lt;", "<").replaceAll("&gt;", ">");
  }
  let c = l.getBBox();
  if (bt(r.htmlLabels)) {
    const h = l.children[0];
    h.style.textAlign = "start";
    const u = et(l);
    c = h.getBoundingClientRect(), u.attr("width", c.width), u.attr("height", c.height);
  }
  return c;
}
p(er, "addText");
async function Xd(e, t, r, i, a = r.class.padding ?? 12) {
  const n = i ? 0 : 3, o = e.insert("g").attr("class", Z(t)).attr("id", t.domId || t.id);
  let s = null, l = null, c = null, h = null, u = 0, f = 0, d = 0;
  if (s = o.insert("g").attr("class", "annotation-group text"), t.annotations.length > 0) {
    const b = t.annotations[0];
    await Hr(s, { text: `«${b}»` }, 0), u = s.node().getBBox().height;
  }
  l = o.insert("g").attr("class", "label-group text"), await Hr(l, t, 0, ["font-weight: bolder"]);
  const g = l.node().getBBox();
  f = g.height, c = o.insert("g").attr("class", "members-group text");
  let m = 0;
  for (const b of t.members) {
    const k = await Hr(c, b, m, [b.parseClassifier()]);
    m += k + n;
  }
  d = c.node().getBBox().height, d <= 0 && (d = a / 2), h = o.insert("g").attr("class", "methods-group text");
  let y = 0;
  for (const b of t.methods) {
    const k = await Hr(h, b, y, [b.parseClassifier()]);
    y += k + n;
  }
  let x = o.node().getBBox();
  if (s !== null) {
    const b = s.node().getBBox();
    s.attr("transform", `translate(${-b.width / 2})`);
  }
  return l.attr("transform", `translate(${-g.width / 2}, ${u})`), x = o.node().getBBox(), c.attr(
    "transform",
    `translate(0, ${u + f + a * 2})`
  ), x = o.node().getBBox(), h.attr(
    "transform",
    `translate(0, ${u + f + (d ? d + a * 4 : a * 2)})`
  ), x = o.node().getBBox(), { shapeSvg: o, bbox: x };
}
p(Xd, "textHelper");
async function Hr(e, t, r, i = []) {
  const a = e.insert("g").attr("class", "label").attr("style", i.join("; ")), n = It();
  let o = "useHtmlLabels" in t ? t.useHtmlLabels : bt(n.htmlLabels) ?? !0, s = "";
  "text" in t ? s = t.text : s = t.label, !o && s.startsWith("\\") && (s = s.substring(1)), yr(s) && (o = !0);
  const l = await Le(
    a,
    La(Ze(s)),
    {
      width: pe(s, n) + 50,
      // Add room for error when splitting text into multiple lines
      classes: "markdown-node-label",
      useHtmlLabels: o
    },
    n
  );
  let c, h = 1;
  if (o) {
    const u = l.children[0], f = et(l);
    h = u.innerHTML.split("<br>").length, u.innerHTML.includes("</math>") && (h += u.innerHTML.split("<mrow>").length - 1);
    const d = u.getElementsByTagName("img");
    if (d) {
      const g = s.replace(/<img[^>]*>/g, "").trim() === "";
      await Promise.all(
        [...d].map(
          (m) => new Promise((y) => {
            function x() {
              var b;
              if (m.style.display = "flex", m.style.flexDirection = "column", g) {
                const k = ((b = n.fontSize) == null ? void 0 : b.toString()) ?? window.getComputedStyle(document.body).fontSize, w = parseInt(k, 10) * 5 + "px";
                m.style.minWidth = w, m.style.maxWidth = w;
              } else
                m.style.width = "100%";
              y(m);
            }
            p(x, "setupImage"), setTimeout(() => {
              m.complete && x();
            }), m.addEventListener("error", x), m.addEventListener("load", x);
          })
        )
      );
    }
    c = u.getBoundingClientRect(), f.attr("width", c.width), f.attr("height", c.height);
  } else {
    i.includes("font-weight: bolder") && et(l).selectAll("tspan").attr("font-weight", ""), h = l.children.length;
    const u = l.children[0];
    (l.textContent === "" || l.textContent.includes("&gt")) && (u.textContent = s[0] + s.substring(1).replaceAll("&gt;", ">").replaceAll("&lt;", "<").trim(), s[1] === " " && (u.textContent = u.textContent[0] + " " + u.textContent.substring(1))), u.textContent === "undefined" && (u.textContent = ""), c = l.getBBox();
  }
  return a.attr("transform", "translate(0," + (-c.height / (2 * h) + r) + ")"), c.height;
}
p(Hr, "addText");
async function Vd(e, t) {
  var R, O;
  const r = at(), i = r.class.padding ?? 12, a = i, n = t.useHtmlLabels ?? bt(r.htmlLabels) ?? !0, o = t;
  o.annotations = o.annotations ?? [], o.members = o.members ?? [], o.methods = o.methods ?? [];
  const { shapeSvg: s, bbox: l } = await Xd(e, t, r, n, a), { labelStyles: c, nodeStyles: h } = Y(t);
  t.labelStyle = c, t.cssStyles = o.styles || "";
  const u = ((R = o.styles) == null ? void 0 : R.join(";")) || h || "";
  t.cssStyles || (t.cssStyles = u.replaceAll("!important", "").split(";"));
  const f = o.members.length === 0 && o.methods.length === 0 && !((O = r.class) != null && O.hideEmptyMembersBox), d = W.svg(s), g = H(t, {});
  t.look !== "handDrawn" && (g.roughness = 0, g.fillStyle = "solid");
  const m = l.width;
  let y = l.height;
  o.members.length === 0 && o.methods.length === 0 ? y += a : o.members.length > 0 && o.methods.length === 0 && (y += a * 2);
  const x = -m / 2, b = -y / 2, k = d.rectangle(
    x - i,
    b - i - (f ? i : o.members.length === 0 && o.methods.length === 0 ? -i / 2 : 0),
    m + 2 * i,
    y + 2 * i + (f ? i * 2 : o.members.length === 0 && o.methods.length === 0 ? -i : 0),
    g
  ), S = s.insert(() => k, ":first-child");
  S.attr("class", "basic label-container");
  const w = S.node().getBBox();
  s.selectAll(".text").each(($, I, D) => {
    var N;
    const B = et(D[I]), M = B.attr("transform");
    let T = 0;
    if (M) {
      const J = RegExp(/translate\(([^,]+),([^)]+)\)/).exec(M);
      J && (T = parseFloat(J[2]));
    }
    let A = T + b + i - (f ? i : o.members.length === 0 && o.methods.length === 0 ? -i / 2 : 0);
    n || (A -= 4);
    let L = x;
    (B.attr("class").includes("label-group") || B.attr("class").includes("annotation-group")) && (L = -((N = B.node()) == null ? void 0 : N.getBBox().width) / 2 || 0, s.selectAll("text").each(function(U, J, lt) {
      window.getComputedStyle(lt[J]).textAnchor === "middle" && (L = 0);
    })), B.attr("transform", `translate(${L}, ${A})`);
  });
  const C = s.select(".annotation-group").node().getBBox().height - (f ? i / 2 : 0) || 0, _ = s.select(".label-group").node().getBBox().height - (f ? i / 2 : 0) || 0, E = s.select(".members-group").node().getBBox().height - (f ? i / 2 : 0) || 0;
  if (o.members.length > 0 || o.methods.length > 0 || f) {
    const $ = d.line(
      w.x,
      C + _ + b + i,
      w.x + w.width,
      C + _ + b + i,
      g
    );
    s.insert(() => $).attr("class", "divider").attr("style", u);
  }
  if (f || o.members.length > 0 || o.methods.length > 0) {
    const $ = d.line(
      w.x,
      C + _ + E + b + a * 2 + i,
      w.x + w.width,
      C + _ + E + b + i + a * 2,
      g
    );
    s.insert(() => $).attr("class", "divider").attr("style", u);
  }
  if (o.look !== "handDrawn" && s.selectAll("path").attr("style", u), S.select(":nth-child(2)").attr("style", u), s.selectAll(".divider").select("path").attr("style", u), t.labelStyle ? s.selectAll("span").attr("style", t.labelStyle) : s.selectAll("span").attr("style", u), !n) {
    const $ = RegExp(/color\s*:\s*([^;]*)/), I = $.exec(u);
    if (I) {
      const D = I[0].replace("color", "fill");
      s.selectAll("tspan").attr("style", D);
    } else if (c) {
      const D = $.exec(c);
      if (D) {
        const B = D[0].replace("color", "fill");
        s.selectAll("tspan").attr("style", B);
      }
    }
  }
  return j(t, S), t.intersect = function($) {
    return q.rect(t, $);
  }, s;
}
p(Vd, "classBox");
async function Zd(e, t) {
  var C, _;
  const { labelStyles: r, nodeStyles: i } = Y(t);
  t.labelStyle = r;
  const a = t, n = t, o = 20, s = 20, l = "verifyMethod" in t, c = Z(t), h = e.insert("g").attr("class", c).attr("id", t.domId ?? t.id);
  let u;
  l ? u = await Kt(
    h,
    `&lt;&lt;${a.type}&gt;&gt;`,
    0,
    t.labelStyle
  ) : u = await Kt(h, "&lt;&lt;Element&gt;&gt;", 0, t.labelStyle);
  let f = u;
  const d = await Kt(
    h,
    a.name,
    f,
    t.labelStyle + "; font-weight: bold;"
  );
  if (f += d + s, l) {
    const E = await Kt(
      h,
      `${a.requirementId ? `id: ${a.requirementId}` : ""}`,
      f,
      t.labelStyle
    );
    f += E;
    const R = await Kt(
      h,
      `${a.text ? `Text: ${a.text}` : ""}`,
      f,
      t.labelStyle
    );
    f += R;
    const O = await Kt(
      h,
      `${a.risk ? `Risk: ${a.risk}` : ""}`,
      f,
      t.labelStyle
    );
    f += O, await Kt(
      h,
      `${a.verifyMethod ? `Verification: ${a.verifyMethod}` : ""}`,
      f,
      t.labelStyle
    );
  } else {
    const E = await Kt(
      h,
      `${n.type ? `Type: ${n.type}` : ""}`,
      f,
      t.labelStyle
    );
    f += E, await Kt(
      h,
      `${n.docRef ? `Doc Ref: ${n.docRef}` : ""}`,
      f,
      t.labelStyle
    );
  }
  const g = (((C = h.node()) == null ? void 0 : C.getBBox().width) ?? 200) + o, m = (((_ = h.node()) == null ? void 0 : _.getBBox().height) ?? 200) + o, y = -g / 2, x = -m / 2, b = W.svg(h), k = H(t, {});
  t.look !== "handDrawn" && (k.roughness = 0, k.fillStyle = "solid");
  const S = b.rectangle(y, x, g, m, k), w = h.insert(() => S, ":first-child");
  if (w.attr("class", "basic label-container").attr("style", i), h.selectAll(".label").each((E, R, O) => {
    const $ = et(O[R]), I = $.attr("transform");
    let D = 0, B = 0;
    if (I) {
      const L = RegExp(/translate\(([^,]+),([^)]+)\)/).exec(I);
      L && (D = parseFloat(L[1]), B = parseFloat(L[2]));
    }
    const M = B - m / 2;
    let T = y + o / 2;
    (R === 0 || R === 1) && (T = D), $.attr("transform", `translate(${T}, ${M + o})`);
  }), f > u + d + s) {
    const E = b.line(
      y,
      x + u + d + s,
      y + g,
      x + u + d + s,
      k
    );
    h.insert(() => E).attr("style", i);
  }
  return j(t, w), t.intersect = function(E) {
    return q.rect(t, E);
  }, h;
}
p(Zd, "requirementBox");
async function Kt(e, t, r, i = "") {
  if (t === "")
    return 0;
  const a = e.insert("g").attr("class", "label").attr("style", i), n = at(), o = n.htmlLabels ?? !0, s = await Le(
    a,
    La(Ze(t)),
    {
      width: pe(t, n) + 50,
      // Add room for error when splitting text into multiple lines
      classes: "markdown-node-label",
      useHtmlLabels: o,
      style: i
    },
    n
  );
  let l;
  if (o) {
    const c = s.children[0], h = et(s);
    l = c.getBoundingClientRect(), h.attr("width", l.width), h.attr("height", l.height);
  } else {
    const c = s.children[0];
    for (const h of c.children)
      h.textContent = h.textContent.replaceAll("&gt;", ">").replaceAll("&lt;", "<"), i && h.setAttribute("style", i);
    l = s.getBBox(), l.height += 6;
  }
  return a.attr("transform", `translate(${-l.width / 2},${-l.height / 2 + r})`), l.height;
}
p(Kt, "addText");
var $w = /* @__PURE__ */ p((e) => {
  switch (e) {
    case "Very High":
      return "red";
    case "High":
      return "orange";
    case "Medium":
      return null;
    case "Low":
      return "blue";
    case "Very Low":
      return "lightblue";
  }
}, "colorFromPriority");
async function Kd(e, t, { config: r }) {
  var I, D;
  const { labelStyles: i, nodeStyles: a } = Y(t);
  t.labelStyle = i || "";
  const n = 10, o = t.width;
  t.width = (t.width ?? 200) - 10;
  const {
    shapeSvg: s,
    bbox: l,
    label: c
  } = await Q(e, t, Z(t)), h = t.padding || 10;
  let u = "", f;
  "ticket" in t && t.ticket && ((I = r == null ? void 0 : r.kanban) != null && I.ticketBaseUrl) && (u = (D = r == null ? void 0 : r.kanban) == null ? void 0 : D.ticketBaseUrl.replace("#TICKET#", t.ticket), f = s.insert("svg:a", ":first-child").attr("class", "kanban-ticket-link").attr("xlink:href", u).attr("target", "_blank"));
  const d = {
    useHtmlLabels: t.useHtmlLabels,
    labelStyle: t.labelStyle || "",
    width: t.width,
    img: t.img,
    padding: t.padding || 8,
    centerLabel: !1
  };
  let g, m;
  f ? { label: g, bbox: m } = await un(
    f,
    "ticket" in t && t.ticket || "",
    d
  ) : { label: g, bbox: m } = await un(
    s,
    "ticket" in t && t.ticket || "",
    d
  );
  const { label: y, bbox: x } = await un(
    s,
    "assigned" in t && t.assigned || "",
    d
  );
  t.width = o;
  const b = 10, k = (t == null ? void 0 : t.width) || 0, S = Math.max(m.height, x.height) / 2, w = Math.max(l.height + b * 2, (t == null ? void 0 : t.height) || 0) + S, C = -k / 2, _ = -w / 2;
  c.attr(
    "transform",
    "translate(" + (h - k / 2) + ", " + (-S - l.height / 2) + ")"
  ), g.attr(
    "transform",
    "translate(" + (h - k / 2) + ", " + (-S + l.height / 2) + ")"
  ), y.attr(
    "transform",
    "translate(" + (h + k / 2 - x.width - 2 * n) + ", " + (-S + l.height / 2) + ")"
  );
  let E;
  const { rx: R, ry: O } = t, { cssStyles: $ } = t;
  if (t.look === "handDrawn") {
    const B = W.svg(s), M = H(t, {}), T = R || O ? B.path(me(C, _, k, w, R || 0), M) : B.rectangle(C, _, k, w, M);
    E = s.insert(() => T, ":first-child"), E.attr("class", "basic label-container").attr("style", $ || null);
  } else {
    E = s.insert("rect", ":first-child"), E.attr("class", "basic label-container __APA__").attr("style", a).attr("rx", R ?? 5).attr("ry", O ?? 5).attr("x", C).attr("y", _).attr("width", k).attr("height", w);
    const B = "priority" in t && t.priority;
    if (B) {
      const M = s.append("line"), T = C + 2, A = _ + Math.floor((R ?? 0) / 2), L = _ + w - Math.floor((R ?? 0) / 2);
      M.attr("x1", T).attr("y1", A).attr("x2", T).attr("y2", L).attr("stroke-width", "4").attr("stroke", $w(B));
    }
  }
  return j(t, E), t.height = w, t.intersect = function(B) {
    return q.rect(t, B);
  }, s;
}
p(Kd, "kanbanItem");
var Aw = [
  {
    semanticName: "Process",
    name: "Rectangle",
    shortName: "rect",
    description: "Standard process shape",
    aliases: ["proc", "process", "rectangle"],
    internalAliases: ["squareRect"],
    handler: Fd
  },
  {
    semanticName: "Event",
    name: "Rounded Rectangle",
    shortName: "rounded",
    description: "Represents an event",
    aliases: ["event"],
    internalAliases: ["roundedRect"],
    handler: Md
  },
  {
    semanticName: "Terminal Point",
    name: "Stadium",
    shortName: "stadium",
    description: "Terminal point",
    aliases: ["terminal", "pill"],
    handler: Ed
  },
  {
    semanticName: "Subprocess",
    name: "Framed Rectangle",
    shortName: "fr-rect",
    description: "Subprocess",
    aliases: ["subprocess", "subproc", "framed-rectangle", "subroutine"],
    handler: Pd
  },
  {
    semanticName: "Database",
    name: "Cylinder",
    shortName: "cyl",
    description: "Database storage",
    aliases: ["db", "database", "cylinder"],
    handler: rd
  },
  {
    semanticName: "Start",
    name: "Circle",
    shortName: "circle",
    description: "Starting point",
    aliases: ["circ"],
    handler: Vf
  },
  {
    semanticName: "Decision",
    name: "Diamond",
    shortName: "diam",
    description: "Decision-making step",
    aliases: ["decision", "diamond", "question"],
    handler: Td
  },
  {
    semanticName: "Prepare Conditional",
    name: "Hexagon",
    shortName: "hex",
    description: "Preparation or condition step",
    aliases: ["hexagon", "prepare"],
    handler: cd
  },
  {
    semanticName: "Data Input/Output",
    name: "Lean Right",
    shortName: "lean-r",
    description: "Represents input or output",
    aliases: ["lean-right", "in-out"],
    internalAliases: ["lean_right"],
    handler: bd
  },
  {
    semanticName: "Data Input/Output",
    name: "Lean Left",
    shortName: "lean-l",
    description: "Represents output or input",
    aliases: ["lean-left", "out-in"],
    internalAliases: ["lean_left"],
    handler: xd
  },
  {
    semanticName: "Priority Action",
    name: "Trapezoid Base Bottom",
    shortName: "trap-b",
    description: "Priority action",
    aliases: ["priority", "trapezoid-bottom", "trapezoid"],
    handler: Wd
  },
  {
    semanticName: "Manual Operation",
    name: "Trapezoid Base Top",
    shortName: "trap-t",
    description: "Represents a manual task",
    aliases: ["manual", "trapezoid-top", "inv-trapezoid"],
    internalAliases: ["inv_trapezoid"],
    handler: md
  },
  {
    semanticName: "Stop",
    name: "Double Circle",
    shortName: "dbl-circ",
    description: "Represents a stop point",
    aliases: ["double-circle"],
    internalAliases: ["doublecircle"],
    handler: ad
  },
  {
    semanticName: "Text Block",
    name: "Text Block",
    shortName: "text",
    description: "Text block",
    handler: zd
  },
  {
    semanticName: "Card",
    name: "Notched Rectangle",
    shortName: "notch-rect",
    description: "Represents a card",
    aliases: ["card", "notched-rectangle"],
    handler: Uf
  },
  {
    semanticName: "Lined/Shaded Process",
    name: "Lined Rectangle",
    shortName: "lin-rect",
    description: "Lined process shape",
    aliases: ["lined-rectangle", "lined-process", "lin-proc", "shaded-process"],
    handler: $d
  },
  {
    semanticName: "Start",
    name: "Small Circle",
    shortName: "sm-circ",
    description: "Small starting point",
    aliases: ["start", "small-circle"],
    internalAliases: ["stateStart"],
    handler: Rd
  },
  {
    semanticName: "Stop",
    name: "Framed Circle",
    shortName: "fr-circ",
    description: "Stop point",
    aliases: ["stop", "framed-circle"],
    internalAliases: ["stateEnd"],
    handler: Dd
  },
  {
    semanticName: "Fork/Join",
    name: "Filled Rectangle",
    shortName: "fork",
    description: "Fork or join in process flow",
    aliases: ["join"],
    internalAliases: ["forkJoin"],
    handler: od
  },
  {
    semanticName: "Collate",
    name: "Hourglass",
    shortName: "hourglass",
    description: "Represents a collate operation",
    aliases: ["hourglass", "collate"],
    handler: hd
  },
  {
    semanticName: "Comment",
    name: "Curly Brace",
    shortName: "brace",
    description: "Adds a comment",
    aliases: ["comment", "brace-l"],
    handler: Qf
  },
  {
    semanticName: "Comment Right",
    name: "Curly Brace",
    shortName: "brace-r",
    description: "Adds a comment",
    handler: Jf
  },
  {
    semanticName: "Comment with braces on both sides",
    name: "Curly Braces",
    shortName: "braces",
    description: "Adds a comment",
    handler: td
  },
  {
    semanticName: "Com Link",
    name: "Lightning Bolt",
    shortName: "bolt",
    description: "Communication link",
    aliases: ["com-link", "lightning-bolt"],
    handler: Cd
  },
  {
    semanticName: "Document",
    name: "Document",
    shortName: "doc",
    description: "Represents a document",
    aliases: ["doc", "document"],
    handler: Yd
  },
  {
    semanticName: "Delay",
    name: "Half-Rounded Rectangle",
    shortName: "delay",
    description: "Represents a delay",
    aliases: ["half-rounded-rectangle"],
    handler: ld
  },
  {
    semanticName: "Direct Access Storage",
    name: "Horizontal Cylinder",
    shortName: "h-cyl",
    description: "Direct access storage",
    aliases: ["das", "horizontal-cylinder"],
    handler: qd
  },
  {
    semanticName: "Disk Storage",
    name: "Lined Cylinder",
    shortName: "lin-cyl",
    description: "Disk storage",
    aliases: ["disk", "lined-cylinder"],
    handler: kd
  },
  {
    semanticName: "Display",
    name: "Curved Trapezoid",
    shortName: "curv-trap",
    description: "Represents a display",
    aliases: ["curved-trapezoid", "display"],
    handler: ed
  },
  {
    semanticName: "Divided Process",
    name: "Divided Rectangle",
    shortName: "div-rect",
    description: "Divided process shape",
    aliases: ["div-proc", "divided-rectangle", "divided-process"],
    handler: id
  },
  {
    semanticName: "Extract",
    name: "Triangle",
    shortName: "tri",
    description: "Extraction process",
    aliases: ["extract", "triangle"],
    handler: jd
  },
  {
    semanticName: "Internal Storage",
    name: "Window Pane",
    shortName: "win-pane",
    description: "Internal storage",
    aliases: ["internal-storage", "window-pane"],
    handler: Ud
  },
  {
    semanticName: "Junction",
    name: "Filled Circle",
    shortName: "f-circ",
    description: "Junction point",
    aliases: ["junction", "filled-circle"],
    handler: nd
  },
  {
    semanticName: "Loop Limit",
    name: "Trapezoidal Pentagon",
    shortName: "notch-pent",
    description: "Loop limit step",
    aliases: ["loop-limit", "notched-pentagon"],
    handler: Hd
  },
  {
    semanticName: "Manual File",
    name: "Flipped Triangle",
    shortName: "flip-tri",
    description: "Manual file operation",
    aliases: ["manual-file", "flipped-triangle"],
    handler: sd
  },
  {
    semanticName: "Manual Input",
    name: "Sloped Rectangle",
    shortName: "sl-rect",
    description: "Manual input step",
    aliases: ["manual-input", "sloped-rectangle"],
    handler: Ad
  },
  {
    semanticName: "Multi-Document",
    name: "Stacked Document",
    shortName: "docs",
    description: "Multiple documents",
    aliases: ["documents", "st-doc", "stacked-document"],
    handler: vd
  },
  {
    semanticName: "Multi-Process",
    name: "Stacked Rectangle",
    shortName: "st-rect",
    description: "Multiple processes",
    aliases: ["procs", "processes", "stacked-rectangle"],
    handler: _d
  },
  {
    semanticName: "Stored Data",
    name: "Bow Tie Rectangle",
    shortName: "bow-rect",
    description: "Stored data",
    aliases: ["stored-data", "bow-tie-rectangle"],
    handler: Gf
  },
  {
    semanticName: "Summary",
    name: "Crossed Circle",
    shortName: "cross-circ",
    description: "Summary",
    aliases: ["summary", "crossed-circle"],
    handler: Kf
  },
  {
    semanticName: "Tagged Document",
    name: "Tagged Document",
    shortName: "tag-doc",
    description: "Tagged document",
    aliases: ["tag-doc", "tagged-document"],
    handler: Nd
  },
  {
    semanticName: "Tagged Process",
    name: "Tagged Rectangle",
    shortName: "tag-rect",
    description: "Tagged process",
    aliases: ["tagged-rectangle", "tag-proc", "tagged-process"],
    handler: Id
  },
  {
    semanticName: "Paper Tape",
    name: "Flag",
    shortName: "flag",
    description: "Paper tape",
    aliases: ["paper-tape"],
    handler: Gd
  },
  {
    semanticName: "Odd",
    name: "Odd",
    shortName: "odd",
    description: "Odd shape",
    internalAliases: ["rect_left_inv_arrow"],
    handler: Bd
  },
  {
    semanticName: "Lined Document",
    name: "Lined Document",
    shortName: "lin-doc",
    description: "Lined document",
    aliases: ["lined-document"],
    handler: wd
  }
], Fw = /* @__PURE__ */ p(() => {
  const t = [
    ...Object.entries({
      // States
      state: Od,
      choice: Xf,
      note: Sd,
      // Rectangles
      rectWithTitle: Ld,
      labelRect: yd,
      // Icons
      iconSquare: pd,
      iconCircle: fd,
      icon: ud,
      iconRounded: dd,
      imageSquare: gd,
      anchor: Yf,
      // Kanban diagram
      kanbanItem: Kd,
      // class diagram
      classBox: Vd,
      // er diagram
      erBox: oo,
      // Requirement diagram
      requirementBox: Zd
    }),
    ...Aw.flatMap((r) => [
      r.shortName,
      ..."aliases" in r ? r.aliases : [],
      ..."internalAliases" in r ? r.internalAliases : []
    ].map((a) => [a, r.handler]))
  ];
  return Object.fromEntries(t);
}, "generateShapeMap"), Qd = Fw();
function Ew(e) {
  return e in Qd;
}
p(Ew, "isValidShape");
var Ua = /* @__PURE__ */ new Map();
async function Jd(e, t, r) {
  let i, a;
  t.shape === "rect" && (t.rx && t.ry ? t.shape = "roundedRect" : t.shape = "squareRect");
  const n = t.shape ? Qd[t.shape] : void 0;
  if (!n)
    throw new Error(`No such shape: ${t.shape}. Please check your syntax.`);
  if (t.link) {
    let o;
    r.config.securityLevel === "sandbox" ? o = "_top" : t.linkTarget && (o = t.linkTarget || "_blank"), i = e.insert("svg:a").attr("xlink:href", t.link).attr("target", o ?? null), a = await n(i, t, r);
  } else
    a = await n(e, t, r), i = a;
  return t.tooltip && a.attr("title", t.tooltip), Ua.set(t.id, i), t.haveCallback && i.attr("class", i.attr("class") + " clickable"), i;
}
p(Jd, "insertNode");
var jT = /* @__PURE__ */ p((e, t) => {
  Ua.set(t.id, e);
}, "setNodeElem"), YT = /* @__PURE__ */ p(() => {
  Ua.clear();
}, "clear"), GT = /* @__PURE__ */ p((e) => {
  const t = Ua.get(e.id);
  F.trace(
    "Transforming node",
    e.diff,
    e,
    "translate(" + (e.x - e.width / 2 - 5) + ", " + e.width / 2 + ")"
  );
  const r = 8, i = e.diff || 0;
  return e.clusterNode ? t.attr(
    "transform",
    "translate(" + (e.x + i - e.width / 2) + ", " + (e.y - e.height / 2 - r) + ")"
  ) : t.attr("transform", "translate(" + e.x + ", " + e.y + ")"), i;
}, "positionNode"), Ow = /* @__PURE__ */ p((e, t, r, i, a, n) => {
  t.arrowTypeStart && Cl(e, "start", t.arrowTypeStart, r, i, a, n), t.arrowTypeEnd && Cl(e, "end", t.arrowTypeEnd, r, i, a, n);
}, "addEdgeMarkers"), Dw = {
  arrow_cross: { type: "cross", fill: !1 },
  arrow_point: { type: "point", fill: !0 },
  arrow_barb: { type: "barb", fill: !0 },
  arrow_circle: { type: "circle", fill: !1 },
  aggregation: { type: "aggregation", fill: !1 },
  extension: { type: "extension", fill: !1 },
  composition: { type: "composition", fill: !0 },
  dependency: { type: "dependency", fill: !0 },
  lollipop: { type: "lollipop", fill: !1 },
  only_one: { type: "onlyOne", fill: !1 },
  zero_or_one: { type: "zeroOrOne", fill: !1 },
  one_or_more: { type: "oneOrMore", fill: !1 },
  zero_or_more: { type: "zeroOrMore", fill: !1 },
  requirement_arrow: { type: "requirement_arrow", fill: !1 },
  requirement_contains: { type: "requirement_contains", fill: !1 }
}, Cl = /* @__PURE__ */ p((e, t, r, i, a, n, o) => {
  var u;
  const s = Dw[r];
  if (!s) {
    F.warn(`Unknown arrow type: ${r}`);
    return;
  }
  const l = s.type, h = `${a}_${n}-${l}${t === "start" ? "Start" : "End"}`;
  if (o && o.trim() !== "") {
    const f = o.replace(/[^\dA-Za-z]/g, "_"), d = `${h}_${f}`;
    if (!document.getElementById(d)) {
      const g = document.getElementById(h);
      if (g) {
        const m = g.cloneNode(!0);
        m.id = d, m.querySelectorAll("path, circle, line").forEach((x) => {
          x.setAttribute("stroke", o), s.fill && x.setAttribute("fill", o);
        }), (u = g.parentNode) == null || u.appendChild(m);
      }
    }
    e.attr(`marker-${t}`, `url(${i}#${d})`);
  } else
    e.attr(`marker-${t}`, `url(${i}#${h})`);
}, "addEdgeMarker"), wa = /* @__PURE__ */ new Map(), _t = /* @__PURE__ */ new Map(), UT = /* @__PURE__ */ p(() => {
  wa.clear(), _t.clear();
}, "clear"), Pr = /* @__PURE__ */ p((e) => e ? e.reduce((r, i) => r + ";" + i, "") : "", "getLabelStyles"), Rw = /* @__PURE__ */ p(async (e, t) => {
  let r = bt(at().flowchart.htmlLabels);
  const i = await Le(e, t.label, {
    style: Pr(t.labelStyle),
    useHtmlLabels: r,
    addSvgBackground: !0,
    isNode: !1
  });
  F.info("abc82", t, t.labelType);
  const a = e.insert("g").attr("class", "edgeLabel"), n = a.insert("g").attr("class", "label");
  n.node().appendChild(i);
  let o = i.getBBox();
  if (r) {
    const l = i.children[0], c = et(i);
    o = l.getBoundingClientRect(), c.attr("width", o.width), c.attr("height", o.height);
  }
  n.attr("transform", "translate(" + -o.width / 2 + ", " + -o.height / 2 + ")"), wa.set(t.id, a), t.width = o.width, t.height = o.height;
  let s;
  if (t.startLabelLeft) {
    const l = await Pe(
      t.startLabelLeft,
      Pr(t.labelStyle)
    ), c = e.insert("g").attr("class", "edgeTerminals"), h = c.insert("g").attr("class", "inner");
    s = h.node().appendChild(l);
    const u = l.getBBox();
    h.attr("transform", "translate(" + -u.width / 2 + ", " + -u.height / 2 + ")"), _t.get(t.id) || _t.set(t.id, {}), _t.get(t.id).startLeft = c, jr(s, t.startLabelLeft);
  }
  if (t.startLabelRight) {
    const l = await Pe(
      t.startLabelRight,
      Pr(t.labelStyle)
    ), c = e.insert("g").attr("class", "edgeTerminals"), h = c.insert("g").attr("class", "inner");
    s = c.node().appendChild(l), h.node().appendChild(l);
    const u = l.getBBox();
    h.attr("transform", "translate(" + -u.width / 2 + ", " + -u.height / 2 + ")"), _t.get(t.id) || _t.set(t.id, {}), _t.get(t.id).startRight = c, jr(s, t.startLabelRight);
  }
  if (t.endLabelLeft) {
    const l = await Pe(t.endLabelLeft, Pr(t.labelStyle)), c = e.insert("g").attr("class", "edgeTerminals"), h = c.insert("g").attr("class", "inner");
    s = h.node().appendChild(l);
    const u = l.getBBox();
    h.attr("transform", "translate(" + -u.width / 2 + ", " + -u.height / 2 + ")"), c.node().appendChild(l), _t.get(t.id) || _t.set(t.id, {}), _t.get(t.id).endLeft = c, jr(s, t.endLabelLeft);
  }
  if (t.endLabelRight) {
    const l = await Pe(t.endLabelRight, Pr(t.labelStyle)), c = e.insert("g").attr("class", "edgeTerminals"), h = c.insert("g").attr("class", "inner");
    s = h.node().appendChild(l);
    const u = l.getBBox();
    h.attr("transform", "translate(" + -u.width / 2 + ", " + -u.height / 2 + ")"), c.node().appendChild(l), _t.get(t.id) || _t.set(t.id, {}), _t.get(t.id).endRight = c, jr(s, t.endLabelRight);
  }
  return i;
}, "insertEdgeLabel");
function jr(e, t) {
  at().flowchart.htmlLabels && e && (e.style.width = t.length * 9 + "px", e.style.height = "12px");
}
p(jr, "setTerminalWidth");
var Pw = /* @__PURE__ */ p((e, t) => {
  F.debug("Moving label abc88 ", e.id, e.label, wa.get(e.id), t);
  let r = t.updatedPath ? t.updatedPath : t.originalPath;
  const i = at(), { subGraphTitleTotalMargin: a } = vs(i);
  if (e.label) {
    const n = wa.get(e.id);
    let o = e.x, s = e.y;
    if (r) {
      const l = Jt.calcLabelPosition(r);
      F.debug(
        "Moving label " + e.label + " from (",
        o,
        ",",
        s,
        ") to (",
        l.x,
        ",",
        l.y,
        ") abc88"
      ), t.updatedPath && (o = l.x, s = l.y);
    }
    n.attr("transform", `translate(${o}, ${s + a / 2})`);
  }
  if (e.startLabelLeft) {
    const n = _t.get(e.id).startLeft;
    let o = e.x, s = e.y;
    if (r) {
      const l = Jt.calcTerminalLabelPosition(e.arrowTypeStart ? 10 : 0, "start_left", r);
      o = l.x, s = l.y;
    }
    n.attr("transform", `translate(${o}, ${s})`);
  }
  if (e.startLabelRight) {
    const n = _t.get(e.id).startRight;
    let o = e.x, s = e.y;
    if (r) {
      const l = Jt.calcTerminalLabelPosition(
        e.arrowTypeStart ? 10 : 0,
        "start_right",
        r
      );
      o = l.x, s = l.y;
    }
    n.attr("transform", `translate(${o}, ${s})`);
  }
  if (e.endLabelLeft) {
    const n = _t.get(e.id).endLeft;
    let o = e.x, s = e.y;
    if (r) {
      const l = Jt.calcTerminalLabelPosition(e.arrowTypeEnd ? 10 : 0, "end_left", r);
      o = l.x, s = l.y;
    }
    n.attr("transform", `translate(${o}, ${s})`);
  }
  if (e.endLabelRight) {
    const n = _t.get(e.id).endRight;
    let o = e.x, s = e.y;
    if (r) {
      const l = Jt.calcTerminalLabelPosition(e.arrowTypeEnd ? 10 : 0, "end_right", r);
      o = l.x, s = l.y;
    }
    n.attr("transform", `translate(${o}, ${s})`);
  }
}, "positionEdgeLabel"), Iw = /* @__PURE__ */ p((e, t) => {
  const r = e.x, i = e.y, a = Math.abs(t.x - r), n = Math.abs(t.y - i), o = e.width / 2, s = e.height / 2;
  return a >= o || n >= s;
}, "outsideNode"), Nw = /* @__PURE__ */ p((e, t, r) => {
  F.debug(`intersection calc abc89:
  outsidePoint: ${JSON.stringify(t)}
  insidePoint : ${JSON.stringify(r)}
  node        : x:${e.x} y:${e.y} w:${e.width} h:${e.height}`);
  const i = e.x, a = e.y, n = Math.abs(i - r.x), o = e.width / 2;
  let s = r.x < t.x ? o - n : o + n;
  const l = e.height / 2, c = Math.abs(t.y - r.y), h = Math.abs(t.x - r.x);
  if (Math.abs(a - t.y) * o > Math.abs(i - t.x) * l) {
    let u = r.y < t.y ? t.y - l - a : a - l - t.y;
    s = h * u / c;
    const f = {
      x: r.x < t.x ? r.x + s : r.x - h + s,
      y: r.y < t.y ? r.y + c - u : r.y - c + u
    };
    return s === 0 && (f.x = t.x, f.y = t.y), h === 0 && (f.x = t.x), c === 0 && (f.y = t.y), F.debug(`abc89 top/bottom calc, Q ${c}, q ${u}, R ${h}, r ${s}`, f), f;
  } else {
    r.x < t.x ? s = t.x - o - i : s = i - o - t.x;
    let u = c * s / h, f = r.x < t.x ? r.x + h - s : r.x - h + s, d = r.y < t.y ? r.y + u : r.y - u;
    return F.debug(`sides calc abc89, Q ${c}, q ${u}, R ${h}, r ${s}`, { _x: f, _y: d }), s === 0 && (f = t.x, d = t.y), h === 0 && (f = t.x), c === 0 && (d = t.y), { x: f, y: d };
  }
}, "intersection"), kl = /* @__PURE__ */ p((e, t) => {
  F.warn("abc88 cutPathAtIntersect", e, t);
  let r = [], i = e[0], a = !1;
  return e.forEach((n) => {
    if (F.info("abc88 checking point", n, t), !Iw(t, n) && !a) {
      const o = Nw(t, i, n);
      F.debug("abc88 inside", n, i, o), F.debug("abc88 intersection", o, t);
      let s = !1;
      r.forEach((l) => {
        s = s || l.x === o.x && l.y === o.y;
      }), r.some((l) => l.x === o.x && l.y === o.y) ? F.warn("abc88 no intersect", o, r) : r.push(o), a = !0;
    } else
      F.warn("abc88 outside", n, i), i = n, a || r.push(n);
  }), F.debug("returning points", r), r;
}, "cutPathAtIntersect");
function tp(e) {
  const t = [], r = [];
  for (let i = 1; i < e.length - 1; i++) {
    const a = e[i - 1], n = e[i], o = e[i + 1];
    (a.x === n.x && n.y === o.y && Math.abs(n.x - o.x) > 5 && Math.abs(n.y - a.y) > 5 || a.y === n.y && n.x === o.x && Math.abs(n.x - a.x) > 5 && Math.abs(n.y - o.y) > 5) && (t.push(n), r.push(i));
  }
  return { cornerPoints: t, cornerPointPositions: r };
}
p(tp, "extractCornerPoints");
var wl = /* @__PURE__ */ p(function(e, t, r) {
  const i = t.x - e.x, a = t.y - e.y, n = Math.sqrt(i * i + a * a), o = r / n;
  return { x: t.x - o * i, y: t.y - o * a };
}, "findAdjacentPoint"), zw = /* @__PURE__ */ p(function(e) {
  const { cornerPointPositions: t } = tp(e), r = [];
  for (let i = 0; i < e.length; i++)
    if (t.includes(i)) {
      const a = e[i - 1], n = e[i + 1], o = e[i], s = wl(a, o, 5), l = wl(n, o, 5), c = l.x - s.x, h = l.y - s.y;
      r.push(s);
      const u = Math.sqrt(2) * 2;
      let f = { x: o.x, y: o.y };
      if (Math.abs(n.x - a.x) > 10 && Math.abs(n.y - a.y) >= 10) {
        F.debug(
          "Corner point fixing",
          Math.abs(n.x - a.x),
          Math.abs(n.y - a.y)
        );
        const d = 5;
        o.x === s.x ? f = {
          x: c < 0 ? s.x - d + u : s.x + d - u,
          y: h < 0 ? s.y - u : s.y + u
        } : f = {
          x: c < 0 ? s.x - u : s.x + u,
          y: h < 0 ? s.y - d + u : s.y + d - u
        };
      } else
        F.debug(
          "Corner point skipping fixing",
          Math.abs(n.x - a.x),
          Math.abs(n.y - a.y)
        );
      r.push(f, l);
    } else
      r.push(e[i]);
  return r;
}, "fixCorners"), qw = /* @__PURE__ */ p(function(e, t, r, i, a, n, o) {
  var R;
  const { handDrawnSeed: s } = at();
  let l = t.points, c = !1;
  const h = a;
  var u = n;
  const f = [];
  for (const O in t.cssCompiledStyles)
    Ih(O) || f.push(t.cssCompiledStyles[O]);
  u.intersect && h.intersect && (l = l.slice(1, t.points.length - 1), l.unshift(h.intersect(l[0])), F.debug(
    "Last point APA12",
    t.start,
    "-->",
    t.end,
    l[l.length - 1],
    u,
    u.intersect(l[l.length - 1])
  ), l.push(u.intersect(l[l.length - 1]))), t.toCluster && (F.info("to cluster abc88", r.get(t.toCluster)), l = kl(t.points, r.get(t.toCluster).node), c = !0), t.fromCluster && (F.debug(
    "from cluster abc88",
    r.get(t.fromCluster),
    JSON.stringify(l, null, 2)
  ), l = kl(l.reverse(), r.get(t.fromCluster).node).reverse(), c = !0);
  let d = l.filter((O) => !Number.isNaN(O.y));
  d = zw(d);
  let g = Ri;
  switch (g = na, t.curve) {
    case "linear":
      g = na;
      break;
    case "basis":
      g = Ri;
      break;
    case "cardinal":
      g = Cu;
      break;
    case "bumpX":
      g = gu;
      break;
    case "bumpY":
      g = mu;
      break;
    case "catmullRom":
      g = wu;
      break;
    case "monotoneX":
      g = Lu;
      break;
    case "monotoneY":
      g = Mu;
      break;
    case "natural":
      g = Au;
      break;
    case "step":
      g = Fu;
      break;
    case "stepAfter":
      g = Ou;
      break;
    case "stepBefore":
      g = Eu;
      break;
    default:
      g = Ri;
  }
  const { x: m, y } = cm(t), x = Pb().x(m).y(y).curve(g);
  let b;
  switch (t.thickness) {
    case "normal":
      b = "edge-thickness-normal";
      break;
    case "thick":
      b = "edge-thickness-thick";
      break;
    case "invisible":
      b = "edge-thickness-invisible";
      break;
    default:
      b = "edge-thickness-normal";
  }
  switch (t.pattern) {
    case "solid":
      b += " edge-pattern-solid";
      break;
    case "dotted":
      b += " edge-pattern-dotted";
      break;
    case "dashed":
      b += " edge-pattern-dashed";
      break;
    default:
      b += " edge-pattern-solid";
  }
  let k, S = x(d);
  const w = Array.isArray(t.style) ? t.style : t.style ? [t.style] : [];
  let C = w.find((O) => O == null ? void 0 : O.startsWith("stroke:"));
  if (t.look === "handDrawn") {
    const O = W.svg(e);
    Object.assign([], d);
    const $ = O.path(S, {
      roughness: 0.3,
      seed: s
    });
    b += " transition", k = et($).select("path").attr("id", t.id).attr("class", " " + b + (t.classes ? " " + t.classes : "")).attr("style", w ? w.reduce((D, B) => D + ";" + B, "") : "");
    let I = k.attr("d");
    k.attr("d", I), e.node().appendChild(k.node());
  } else {
    const O = f.join(";"), $ = w ? w.reduce((B, M) => B + M + ";", "") : "";
    let I = "";
    t.animate && (I = " edge-animation-fast"), t.animation && (I = " edge-animation-" + t.animation);
    const D = O ? O + ";" + $ + ";" : $;
    k = e.append("path").attr("d", S).attr("id", t.id).attr(
      "class",
      " " + b + (t.classes ? " " + t.classes : "") + (I ?? "")
    ).attr("style", D), C = (R = D.match(/stroke:([^;]+)/)) == null ? void 0 : R[1];
  }
  let _ = "";
  (at().flowchart.arrowMarkerAbsolute || at().state.arrowMarkerAbsolute) && (_ = tc(!0)), F.info("arrowTypeStart", t.arrowTypeStart), F.info("arrowTypeEnd", t.arrowTypeEnd), Ow(k, t, _, o, i, C);
  let E = {};
  return c && (E.updatedPath = l), E.originalPath = t.points, E;
}, "insertEdge"), Ww = /* @__PURE__ */ p((e, t, r, i) => {
  t.forEach((a) => {
    a_[a](e, r, i);
  });
}, "insertMarkers"), Hw = /* @__PURE__ */ p((e, t, r) => {
  F.trace("Making markers for ", r), e.append("defs").append("marker").attr("id", r + "_" + t + "-extensionStart").attr("class", "marker extension " + t).attr("refX", 18).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("path").attr("d", "M 1,7 L18,13 V 1 Z"), e.append("defs").append("marker").attr("id", r + "_" + t + "-extensionEnd").attr("class", "marker extension " + t).attr("refX", 1).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 28).attr("orient", "auto").append("path").attr("d", "M 1,1 V 13 L18,7 Z");
}, "extension"), jw = /* @__PURE__ */ p((e, t, r) => {
  e.append("defs").append("marker").attr("id", r + "_" + t + "-compositionStart").attr("class", "marker composition " + t).attr("refX", 18).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L1,7 L9,1 Z"), e.append("defs").append("marker").attr("id", r + "_" + t + "-compositionEnd").attr("class", "marker composition " + t).attr("refX", 1).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 28).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L1,7 L9,1 Z");
}, "composition"), Yw = /* @__PURE__ */ p((e, t, r) => {
  e.append("defs").append("marker").attr("id", r + "_" + t + "-aggregationStart").attr("class", "marker aggregation " + t).attr("refX", 18).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L1,7 L9,1 Z"), e.append("defs").append("marker").attr("id", r + "_" + t + "-aggregationEnd").attr("class", "marker aggregation " + t).attr("refX", 1).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 28).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L1,7 L9,1 Z");
}, "aggregation"), Gw = /* @__PURE__ */ p((e, t, r) => {
  e.append("defs").append("marker").attr("id", r + "_" + t + "-dependencyStart").attr("class", "marker dependency " + t).attr("refX", 6).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("path").attr("d", "M 5,7 L9,13 L1,7 L9,1 Z"), e.append("defs").append("marker").attr("id", r + "_" + t + "-dependencyEnd").attr("class", "marker dependency " + t).attr("refX", 13).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 28).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L14,7 L9,1 Z");
}, "dependency"), Uw = /* @__PURE__ */ p((e, t, r) => {
  e.append("defs").append("marker").attr("id", r + "_" + t + "-lollipopStart").attr("class", "marker lollipop " + t).attr("refX", 13).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("circle").attr("stroke", "black").attr("fill", "transparent").attr("cx", 7).attr("cy", 7).attr("r", 6), e.append("defs").append("marker").attr("id", r + "_" + t + "-lollipopEnd").attr("class", "marker lollipop " + t).attr("refX", 1).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("circle").attr("stroke", "black").attr("fill", "transparent").attr("cx", 7).attr("cy", 7).attr("r", 6);
}, "lollipop"), Xw = /* @__PURE__ */ p((e, t, r) => {
  e.append("marker").attr("id", r + "_" + t + "-pointEnd").attr("class", "marker " + t).attr("viewBox", "0 0 10 10").attr("refX", 5).attr("refY", 5).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 8).attr("markerHeight", 8).attr("orient", "auto").append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("class", "arrowMarkerPath").style("stroke-width", 1).style("stroke-dasharray", "1,0"), e.append("marker").attr("id", r + "_" + t + "-pointStart").attr("class", "marker " + t).attr("viewBox", "0 0 10 10").attr("refX", 4.5).attr("refY", 5).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 8).attr("markerHeight", 8).attr("orient", "auto").append("path").attr("d", "M 0 5 L 10 10 L 10 0 z").attr("class", "arrowMarkerPath").style("stroke-width", 1).style("stroke-dasharray", "1,0");
}, "point"), Vw = /* @__PURE__ */ p((e, t, r) => {
  e.append("marker").attr("id", r + "_" + t + "-circleEnd").attr("class", "marker " + t).attr("viewBox", "0 0 10 10").attr("refX", 11).attr("refY", 5).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto").append("circle").attr("cx", "5").attr("cy", "5").attr("r", "5").attr("class", "arrowMarkerPath").style("stroke-width", 1).style("stroke-dasharray", "1,0"), e.append("marker").attr("id", r + "_" + t + "-circleStart").attr("class", "marker " + t).attr("viewBox", "0 0 10 10").attr("refX", -1).attr("refY", 5).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto").append("circle").attr("cx", "5").attr("cy", "5").attr("r", "5").attr("class", "arrowMarkerPath").style("stroke-width", 1).style("stroke-dasharray", "1,0");
}, "circle"), Zw = /* @__PURE__ */ p((e, t, r) => {
  e.append("marker").attr("id", r + "_" + t + "-crossEnd").attr("class", "marker cross " + t).attr("viewBox", "0 0 11 11").attr("refX", 12).attr("refY", 5.2).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto").append("path").attr("d", "M 1,1 l 9,9 M 10,1 l -9,9").attr("class", "arrowMarkerPath").style("stroke-width", 2).style("stroke-dasharray", "1,0"), e.append("marker").attr("id", r + "_" + t + "-crossStart").attr("class", "marker cross " + t).attr("viewBox", "0 0 11 11").attr("refX", -1).attr("refY", 5.2).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto").append("path").attr("d", "M 1,1 l 9,9 M 10,1 l -9,9").attr("class", "arrowMarkerPath").style("stroke-width", 2).style("stroke-dasharray", "1,0");
}, "cross"), Kw = /* @__PURE__ */ p((e, t, r) => {
  e.append("defs").append("marker").attr("id", r + "_" + t + "-barbEnd").attr("refX", 19).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 14).attr("markerUnits", "userSpaceOnUse").attr("orient", "auto").append("path").attr("d", "M 19,7 L9,13 L14,7 L9,1 Z");
}, "barb"), Qw = /* @__PURE__ */ p((e, t, r) => {
  e.append("defs").append("marker").attr("id", r + "_" + t + "-onlyOneStart").attr("class", "marker onlyOne " + t).attr("refX", 0).attr("refY", 9).attr("markerWidth", 18).attr("markerHeight", 18).attr("orient", "auto").append("path").attr("d", "M9,0 L9,18 M15,0 L15,18"), e.append("defs").append("marker").attr("id", r + "_" + t + "-onlyOneEnd").attr("class", "marker onlyOne " + t).attr("refX", 18).attr("refY", 9).attr("markerWidth", 18).attr("markerHeight", 18).attr("orient", "auto").append("path").attr("d", "M3,0 L3,18 M9,0 L9,18");
}, "only_one"), Jw = /* @__PURE__ */ p((e, t, r) => {
  const i = e.append("defs").append("marker").attr("id", r + "_" + t + "-zeroOrOneStart").attr("class", "marker zeroOrOne " + t).attr("refX", 0).attr("refY", 9).attr("markerWidth", 30).attr("markerHeight", 18).attr("orient", "auto");
  i.append("circle").attr("fill", "white").attr("cx", 21).attr("cy", 9).attr("r", 6), i.append("path").attr("d", "M9,0 L9,18");
  const a = e.append("defs").append("marker").attr("id", r + "_" + t + "-zeroOrOneEnd").attr("class", "marker zeroOrOne " + t).attr("refX", 30).attr("refY", 9).attr("markerWidth", 30).attr("markerHeight", 18).attr("orient", "auto");
  a.append("circle").attr("fill", "white").attr("cx", 9).attr("cy", 9).attr("r", 6), a.append("path").attr("d", "M21,0 L21,18");
}, "zero_or_one"), t_ = /* @__PURE__ */ p((e, t, r) => {
  e.append("defs").append("marker").attr("id", r + "_" + t + "-oneOrMoreStart").attr("class", "marker oneOrMore " + t).attr("refX", 18).attr("refY", 18).attr("markerWidth", 45).attr("markerHeight", 36).attr("orient", "auto").append("path").attr("d", "M0,18 Q 18,0 36,18 Q 18,36 0,18 M42,9 L42,27"), e.append("defs").append("marker").attr("id", r + "_" + t + "-oneOrMoreEnd").attr("class", "marker oneOrMore " + t).attr("refX", 27).attr("refY", 18).attr("markerWidth", 45).attr("markerHeight", 36).attr("orient", "auto").append("path").attr("d", "M3,9 L3,27 M9,18 Q27,0 45,18 Q27,36 9,18");
}, "one_or_more"), e_ = /* @__PURE__ */ p((e, t, r) => {
  const i = e.append("defs").append("marker").attr("id", r + "_" + t + "-zeroOrMoreStart").attr("class", "marker zeroOrMore " + t).attr("refX", 18).attr("refY", 18).attr("markerWidth", 57).attr("markerHeight", 36).attr("orient", "auto");
  i.append("circle").attr("fill", "white").attr("cx", 48).attr("cy", 18).attr("r", 6), i.append("path").attr("d", "M0,18 Q18,0 36,18 Q18,36 0,18");
  const a = e.append("defs").append("marker").attr("id", r + "_" + t + "-zeroOrMoreEnd").attr("class", "marker zeroOrMore " + t).attr("refX", 39).attr("refY", 18).attr("markerWidth", 57).attr("markerHeight", 36).attr("orient", "auto");
  a.append("circle").attr("fill", "white").attr("cx", 9).attr("cy", 18).attr("r", 6), a.append("path").attr("d", "M21,18 Q39,0 57,18 Q39,36 21,18");
}, "zero_or_more"), r_ = /* @__PURE__ */ p((e, t, r) => {
  e.append("defs").append("marker").attr("id", r + "_" + t + "-requirement_arrowEnd").attr("refX", 20).attr("refY", 10).attr("markerWidth", 20).attr("markerHeight", 20).attr("orient", "auto").append("path").attr(
    "d",
    `M0,0
      L20,10
      M20,10
      L0,20`
  );
}, "requirement_arrow"), i_ = /* @__PURE__ */ p((e, t, r) => {
  const i = e.append("defs").append("marker").attr("id", r + "_" + t + "-requirement_containsStart").attr("refX", 0).attr("refY", 10).attr("markerWidth", 20).attr("markerHeight", 20).attr("orient", "auto").append("g");
  i.append("circle").attr("cx", 10).attr("cy", 10).attr("r", 9).attr("fill", "none"), i.append("line").attr("x1", 1).attr("x2", 19).attr("y1", 10).attr("y2", 10), i.append("line").attr("y1", 1).attr("y2", 19).attr("x1", 10).attr("x2", 10);
}, "requirement_contains"), a_ = {
  extension: Hw,
  composition: jw,
  aggregation: Yw,
  dependency: Gw,
  lollipop: Uw,
  point: Xw,
  circle: Vw,
  cross: Zw,
  barb: Kw,
  only_one: Qw,
  zero_or_one: Jw,
  one_or_more: t_,
  zero_or_more: e_,
  requirement_arrow: r_,
  requirement_contains: i_
}, n_ = Ww, s_ = {
  common: vr,
  getConfig: It,
  insertCluster: pw,
  insertEdge: qw,
  insertEdgeLabel: Rw,
  insertMarkers: n_,
  insertNode: Jd,
  interpolateToCurve: Ws,
  labelHelper: Q,
  log: F,
  positionEdgeLabel: Pw
}, oi = {}, ep = /* @__PURE__ */ p((e) => {
  for (const t of e)
    oi[t.name] = t;
}, "registerLayoutLoaders"), o_ = /* @__PURE__ */ p(() => {
  ep([
    {
      name: "dagre",
      loader: /* @__PURE__ */ p(async () => await import("./dagre-JOIXM2OF-DS02vJJm.js"), "loader")
    }
  ]);
}, "registerDefaultLayoutLoaders");
o_();
var XT = /* @__PURE__ */ p(async (e, t) => {
  if (!(e.layoutAlgorithm in oi))
    throw new Error(`Unknown layout algorithm: ${e.layoutAlgorithm}`);
  const r = oi[e.layoutAlgorithm];
  return (await r.loader()).render(e, t, s_, {
    algorithm: r.algorithm
  });
}, "render"), VT = /* @__PURE__ */ p((e = "", { fallback: t = "dagre" } = {}) => {
  if (e in oi)
    return e;
  if (t in oi)
    return F.warn(`Layout algorithm ${e} is not registered. Using ${t} as fallback.`), t;
  throw new Error(`Both layout algorithms ${e} and ${t} are not registered.`);
}, "getRegisteredLayoutAlgorithm"), _l = {
  name: "mermaid",
  version: "11.9.0",
  description: "Markdown-ish syntax for generating flowcharts, mindmaps, sequence diagrams, class diagrams, gantt charts, git graphs and more.",
  type: "module",
  module: "./dist/mermaid.core.mjs",
  types: "./dist/mermaid.d.ts",
  exports: {
    ".": {
      types: "./dist/mermaid.d.ts",
      import: "./dist/mermaid.core.mjs",
      default: "./dist/mermaid.core.mjs"
    },
    "./*": "./*"
  },
  keywords: [
    "diagram",
    "markdown",
    "flowchart",
    "sequence diagram",
    "gantt",
    "class diagram",
    "git graph",
    "mindmap",
    "packet diagram",
    "c4 diagram",
    "er diagram",
    "pie chart",
    "pie diagram",
    "quadrant chart",
    "requirement diagram",
    "graph"
  ],
  scripts: {
    clean: "rimraf dist",
    dev: "pnpm -w dev",
    "docs:code": "typedoc src/defaultConfig.ts src/config.ts src/mermaid.ts && prettier --write ./src/docs/config/setup",
    "docs:build": "rimraf ../../docs && pnpm docs:code && pnpm docs:spellcheck && tsx scripts/docs.cli.mts",
    "docs:verify": "pnpm docs:code && pnpm docs:spellcheck && tsx scripts/docs.cli.mts --verify",
    "docs:pre:vitepress": "pnpm --filter ./src/docs prefetch && rimraf src/vitepress && pnpm docs:code && tsx scripts/docs.cli.mts --vitepress && pnpm --filter ./src/vitepress install --no-frozen-lockfile --ignore-scripts",
    "docs:build:vitepress": "pnpm docs:pre:vitepress && (cd src/vitepress && pnpm run build) && cpy --flat src/docs/landing/ ./src/vitepress/.vitepress/dist/landing",
    "docs:dev": 'pnpm docs:pre:vitepress && concurrently "pnpm --filter ./src/vitepress dev" "tsx scripts/docs.cli.mts --watch --vitepress"',
    "docs:dev:docker": 'pnpm docs:pre:vitepress && concurrently "pnpm --filter ./src/vitepress dev:docker" "tsx scripts/docs.cli.mts --watch --vitepress"',
    "docs:serve": "pnpm docs:build:vitepress && vitepress serve src/vitepress",
    "docs:spellcheck": 'cspell "src/docs/**/*.md"',
    "docs:release-version": "tsx scripts/update-release-version.mts",
    "docs:verify-version": "tsx scripts/update-release-version.mts --verify",
    "types:build-config": "tsx scripts/create-types-from-json-schema.mts",
    "types:verify-config": "tsx scripts/create-types-from-json-schema.mts --verify",
    checkCircle: "npx madge --circular ./src",
    prepublishOnly: "pnpm docs:verify-version"
  },
  repository: {
    type: "git",
    url: "https://github.com/mermaid-js/mermaid"
  },
  author: "Knut Sveidqvist",
  license: "MIT",
  standard: {
    ignore: [
      "**/parser/*.js",
      "dist/**/*.js",
      "cypress/**/*.js"
    ],
    globals: [
      "page"
    ]
  },
  dependencies: {
    "@braintree/sanitize-url": "^7.0.4",
    "@iconify/utils": "^2.1.33",
    "@mermaid-js/parser": "workspace:^",
    "@types/d3": "^7.4.3",
    cytoscape: "^3.29.3",
    "cytoscape-cose-bilkent": "^4.1.0",
    "cytoscape-fcose": "^2.2.0",
    d3: "^7.9.0",
    "d3-sankey": "^0.12.3",
    "dagre-d3-es": "7.0.11",
    dayjs: "^1.11.13",
    dompurify: "^3.2.5",
    katex: "^0.16.22",
    khroma: "^2.1.0",
    "lodash-es": "^4.17.21",
    marked: "^16.0.0",
    roughjs: "^4.6.6",
    stylis: "^4.3.6",
    "ts-dedent": "^2.2.0",
    uuid: "^11.1.0"
  },
  devDependencies: {
    "@adobe/jsonschema2md": "^8.0.2",
    "@iconify/types": "^2.0.0",
    "@types/cytoscape": "^3.21.9",
    "@types/cytoscape-fcose": "^2.2.4",
    "@types/d3-sankey": "^0.12.4",
    "@types/d3-scale": "^4.0.9",
    "@types/d3-scale-chromatic": "^3.1.0",
    "@types/d3-selection": "^3.0.11",
    "@types/d3-shape": "^3.1.7",
    "@types/jsdom": "^21.1.7",
    "@types/katex": "^0.16.7",
    "@types/lodash-es": "^4.17.12",
    "@types/micromatch": "^4.0.9",
    "@types/stylis": "^4.2.7",
    "@types/uuid": "^10.0.0",
    ajv: "^8.17.1",
    canvas: "^3.1.0",
    chokidar: "3.6.0",
    concurrently: "^9.1.2",
    "csstree-validator": "^4.0.1",
    globby: "^14.0.2",
    jison: "^0.4.18",
    "js-base64": "^3.7.7",
    jsdom: "^26.1.0",
    "json-schema-to-typescript": "^15.0.4",
    micromatch: "^4.0.8",
    "path-browserify": "^1.0.1",
    prettier: "^3.5.2",
    remark: "^15.0.1",
    "remark-frontmatter": "^5.0.0",
    "remark-gfm": "^4.0.1",
    rimraf: "^6.0.1",
    "start-server-and-test": "^2.0.10",
    "type-fest": "^4.35.0",
    typedoc: "^0.27.8",
    "typedoc-plugin-markdown": "^4.4.2",
    typescript: "~5.7.3",
    "unist-util-flatmap": "^1.0.0",
    "unist-util-visit": "^5.0.0",
    vitepress: "^1.0.2",
    "vitepress-plugin-search": "1.0.4-alpha.22"
  },
  files: [
    "dist/",
    "README.md"
  ],
  publishConfig: {
    access: "public"
  }
}, l_ = /* @__PURE__ */ p((e) => {
  var a;
  const { securityLevel: t } = at();
  let r = et("body");
  if (t === "sandbox") {
    const o = ((a = et(`#i${e}`).node()) == null ? void 0 : a.contentDocument) ?? document;
    r = et(o.body);
  }
  return r.select(`#${e}`);
}, "selectSvgElement"), rp = "comm", ip = "rule", ap = "decl", c_ = "@import", h_ = "@namespace", u_ = "@keyframes", f_ = "@layer", np = Math.abs, lo = String.fromCharCode;
function sp(e) {
  return e.trim();
}
function zi(e, t, r) {
  return e.replace(t, r);
}
function d_(e, t, r) {
  return e.indexOf(t, r);
}
function nr(e, t) {
  return e.charCodeAt(t) | 0;
}
function wr(e, t, r) {
  return e.slice(t, r);
}
function Qt(e) {
  return e.length;
}
function p_(e) {
  return e.length;
}
function Li(e, t) {
  return t.push(e), e;
}
var Xa = 1, _r = 1, op = 0, jt = 0, mt = 0, Mr = "";
function co(e, t, r, i, a, n, o, s) {
  return { value: e, root: t, parent: r, type: i, props: a, children: n, line: Xa, column: _r, length: o, return: "", siblings: s };
}
function g_() {
  return mt;
}
function m_() {
  return mt = jt > 0 ? nr(Mr, --jt) : 0, _r--, mt === 10 && (_r = 1, Xa--), mt;
}
function Ut() {
  return mt = jt < op ? nr(Mr, jt++) : 0, _r++, mt === 10 && (_r = 1, Xa++), mt;
}
function ke() {
  return nr(Mr, jt);
}
function qi() {
  return jt;
}
function Va(e, t) {
  return wr(Mr, e, t);
}
function li(e) {
  switch (e) {
    case 0:
    case 9:
    case 10:
    case 13:
    case 32:
      return 5;
    case 33:
    case 43:
    case 44:
    case 47:
    case 62:
    case 64:
    case 126:
    case 59:
    case 123:
    case 125:
      return 4;
    case 58:
      return 3;
    case 34:
    case 39:
    case 40:
    case 91:
      return 2;
    case 41:
    case 93:
      return 1;
  }
  return 0;
}
function y_(e) {
  return Xa = _r = 1, op = Qt(Mr = e), jt = 0, [];
}
function x_(e) {
  return Mr = "", e;
}
function fn(e) {
  return sp(Va(jt - 1, rs(e === 91 ? e + 2 : e === 40 ? e + 1 : e)));
}
function b_(e) {
  for (; (mt = ke()) && mt < 33; )
    Ut();
  return li(e) > 2 || li(mt) > 3 ? "" : " ";
}
function C_(e, t) {
  for (; --t && Ut() && !(mt < 48 || mt > 102 || mt > 57 && mt < 65 || mt > 70 && mt < 97); )
    ;
  return Va(e, qi() + (t < 6 && ke() == 32 && Ut() == 32));
}
function rs(e) {
  for (; Ut(); )
    switch (mt) {
      case e:
        return jt;
      case 34:
      case 39:
        e !== 34 && e !== 39 && rs(mt);
        break;
      case 40:
        e === 41 && rs(e);
        break;
      case 92:
        Ut();
        break;
    }
  return jt;
}
function k_(e, t) {
  for (; Ut() && e + mt !== 57; )
    if (e + mt === 84 && ke() === 47)
      break;
  return "/*" + Va(t, jt - 1) + "*" + lo(e === 47 ? e : Ut());
}
function w_(e) {
  for (; !li(ke()); )
    Ut();
  return Va(e, jt);
}
function __(e) {
  return x_(Wi("", null, null, null, [""], e = y_(e), 0, [0], e));
}
function Wi(e, t, r, i, a, n, o, s, l) {
  for (var c = 0, h = 0, u = o, f = 0, d = 0, g = 0, m = 1, y = 1, x = 1, b = 0, k = "", S = a, w = n, C = i, _ = k; y; )
    switch (g = b, b = Ut()) {
      case 40:
        if (g != 108 && nr(_, u - 1) == 58) {
          d_(_ += zi(fn(b), "&", "&\f"), "&\f", np(c ? s[c - 1] : 0)) != -1 && (x = -1);
          break;
        }
      case 34:
      case 39:
      case 91:
        _ += fn(b);
        break;
      case 9:
      case 10:
      case 13:
      case 32:
        _ += b_(g);
        break;
      case 92:
        _ += C_(qi() - 1, 7);
        continue;
      case 47:
        switch (ke()) {
          case 42:
          case 47:
            Li(v_(k_(Ut(), qi()), t, r, l), l), (li(g || 1) == 5 || li(ke() || 1) == 5) && Qt(_) && wr(_, -1, void 0) !== " " && (_ += " ");
            break;
          default:
            _ += "/";
        }
        break;
      case 123 * m:
        s[c++] = Qt(_) * x;
      case 125 * m:
      case 59:
      case 0:
        switch (b) {
          case 0:
          case 125:
            y = 0;
          case 59 + h:
            x == -1 && (_ = zi(_, /\f/g, "")), d > 0 && (Qt(_) - u || m === 0 && g === 47) && Li(d > 32 ? Sl(_ + ";", i, r, u - 1, l) : Sl(zi(_, " ", "") + ";", i, r, u - 2, l), l);
            break;
          case 59:
            _ += ";";
          default:
            if (Li(C = vl(_, t, r, c, h, a, s, k, S = [], w = [], u, n), n), b === 123)
              if (h === 0)
                Wi(_, t, C, C, S, n, u, s, w);
              else {
                switch (f) {
                  case 99:
                    if (nr(_, 3) === 110) break;
                  case 108:
                    if (nr(_, 2) === 97) break;
                  default:
                    h = 0;
                  case 100:
                  case 109:
                  case 115:
                }
                h ? Wi(e, C, C, i && Li(vl(e, C, C, 0, 0, a, s, k, a, S = [], u, w), w), a, w, u, s, i ? S : w) : Wi(_, C, C, C, [""], w, 0, s, w);
              }
        }
        c = h = d = 0, m = x = 1, k = _ = "", u = o;
        break;
      case 58:
        u = 1 + Qt(_), d = g;
      default:
        if (m < 1) {
          if (b == 123)
            --m;
          else if (b == 125 && m++ == 0 && m_() == 125)
            continue;
        }
        switch (_ += lo(b), b * m) {
          case 38:
            x = h > 0 ? 1 : (_ += "\f", -1);
            break;
          case 44:
            s[c++] = (Qt(_) - 1) * x, x = 1;
            break;
          case 64:
            ke() === 45 && (_ += fn(Ut())), f = ke(), h = u = Qt(k = _ += w_(qi())), b++;
            break;
          case 45:
            g === 45 && Qt(_) == 2 && (m = 0);
        }
    }
  return n;
}
function vl(e, t, r, i, a, n, o, s, l, c, h, u) {
  for (var f = a - 1, d = a === 0 ? n : [""], g = p_(d), m = 0, y = 0, x = 0; m < i; ++m)
    for (var b = 0, k = wr(e, f + 1, f = np(y = o[m])), S = e; b < g; ++b)
      (S = sp(y > 0 ? d[b] + " " + k : zi(k, /&\f/g, d[b]))) && (l[x++] = S);
  return co(e, t, r, a === 0 ? ip : s, l, c, h, u);
}
function v_(e, t, r, i) {
  return co(e, t, r, rp, lo(g_()), wr(e, 2, -2), 0, i);
}
function Sl(e, t, r, i, a) {
  return co(e, t, r, ap, wr(e, 0, i), wr(e, i + 1, -1), i, a);
}
function is(e, t) {
  for (var r = "", i = 0; i < e.length; i++)
    r += t(e[i], i, e, t) || "";
  return r;
}
function S_(e, t, r, i) {
  switch (e.type) {
    case f_:
      if (e.children.length) break;
    case c_:
    case h_:
    case ap:
      return e.return = e.return || e.value;
    case rp:
      return "";
    case u_:
      return e.return = e.value + "{" + is(e.children, i) + "}";
    case ip:
      if (!Qt(e.value = e.props.join(","))) return "";
  }
  return Qt(r = is(e.children, i)) ? e.return = e.value + "{" + r + "}" : "";
}
var T_ = Iu(Object.keys, Object), B_ = Object.prototype, L_ = B_.hasOwnProperty;
function M_(e) {
  if (!Na(e))
    return T_(e);
  var t = [];
  for (var r in Object(e))
    L_.call(e, r) && r != "constructor" && t.push(r);
  return t;
}
var as = Ve(ae, "DataView"), ns = Ve(ae, "Promise"), ss = Ve(ae, "Set"), os = Ve(ae, "WeakMap"), Tl = "[object Map]", $_ = "[object Object]", Bl = "[object Promise]", Ll = "[object Set]", Ml = "[object WeakMap]", $l = "[object DataView]", A_ = Xe(as), F_ = Xe(si), E_ = Xe(ns), O_ = Xe(ss), D_ = Xe(os), Fe = Tr;
(as && Fe(new as(new ArrayBuffer(1))) != $l || si && Fe(new si()) != Tl || ns && Fe(ns.resolve()) != Bl || ss && Fe(new ss()) != Ll || os && Fe(new os()) != Ml) && (Fe = function(e) {
  var t = Tr(e), r = t == $_ ? e.constructor : void 0, i = r ? Xe(r) : "";
  if (i)
    switch (i) {
      case A_:
        return $l;
      case F_:
        return Tl;
      case E_:
        return Bl;
      case O_:
        return Ll;
      case D_:
        return Ml;
    }
  return t;
});
var R_ = "[object Map]", P_ = "[object Set]", I_ = Object.prototype, N_ = I_.hasOwnProperty;
function Al(e) {
  if (e == null)
    return !0;
  if (za(e) && (fa(e) || typeof e == "string" || typeof e.splice == "function" || zs(e) || qs(e) || ua(e)))
    return !e.length;
  var t = Fe(e);
  if (t == R_ || t == P_)
    return !e.size;
  if (Na(e))
    return !M_(e).length;
  for (var r in e)
    if (N_.call(e, r))
      return !1;
  return !0;
}
var lp = "c4", z_ = /* @__PURE__ */ p((e) => /^\s*C4Context|C4Container|C4Component|C4Dynamic|C4Deployment/.test(e), "detector"), q_ = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./c4Diagram-6F6E4RAY-C2Q3PLas.js");
  return { id: lp, diagram: e };
}, "loader"), W_ = {
  id: lp,
  detector: z_,
  loader: q_
}, H_ = W_, cp = "flowchart", j_ = /* @__PURE__ */ p((e, t) => {
  var r, i;
  return ((r = t == null ? void 0 : t.flowchart) == null ? void 0 : r.defaultRenderer) === "dagre-wrapper" || ((i = t == null ? void 0 : t.flowchart) == null ? void 0 : i.defaultRenderer) === "elk" ? !1 : /^\s*graph/.test(e);
}, "detector"), Y_ = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./flowDiagram-KYDEHFYC-B_9iElls.js");
  return { id: cp, diagram: e };
}, "loader"), G_ = {
  id: cp,
  detector: j_,
  loader: Y_
}, U_ = G_, hp = "flowchart-v2", X_ = /* @__PURE__ */ p((e, t) => {
  var r, i, a;
  return ((r = t == null ? void 0 : t.flowchart) == null ? void 0 : r.defaultRenderer) === "dagre-d3" ? !1 : (((i = t == null ? void 0 : t.flowchart) == null ? void 0 : i.defaultRenderer) === "elk" && (t.layout = "elk"), /^\s*graph/.test(e) && ((a = t == null ? void 0 : t.flowchart) == null ? void 0 : a.defaultRenderer) === "dagre-wrapper" ? !0 : /^\s*flowchart/.test(e));
}, "detector"), V_ = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./flowDiagram-KYDEHFYC-B_9iElls.js");
  return { id: hp, diagram: e };
}, "loader"), Z_ = {
  id: hp,
  detector: X_,
  loader: V_
}, K_ = Z_, up = "er", Q_ = /* @__PURE__ */ p((e) => /^\s*erDiagram/.test(e), "detector"), J_ = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./erDiagram-3M52JZNH-2cTNqbHu.js");
  return { id: up, diagram: e };
}, "loader"), tv = {
  id: up,
  detector: Q_,
  loader: J_
}, ev = tv, fp = "gitGraph", rv = /* @__PURE__ */ p((e) => /^\s*gitGraph/.test(e), "detector"), iv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./gitGraphDiagram-GW3U2K7C-C7fQUoAq.js");
  return { id: fp, diagram: e };
}, "loader"), av = {
  id: fp,
  detector: rv,
  loader: iv
}, nv = av, dp = "gantt", sv = /* @__PURE__ */ p((e) => /^\s*gantt/.test(e), "detector"), ov = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./ganttDiagram-EK5VF46D-CmyVtBsd.js");
  return { id: dp, diagram: e };
}, "loader"), lv = {
  id: dp,
  detector: sv,
  loader: ov
}, cv = lv, pp = "info", hv = /* @__PURE__ */ p((e) => /^\s*info/.test(e), "detector"), uv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./infoDiagram-LHK5PUON-LRq4hgkp.js");
  return { id: pp, diagram: e };
}, "loader"), fv = {
  id: pp,
  detector: hv,
  loader: uv
}, gp = "pie", dv = /* @__PURE__ */ p((e) => /^\s*pie/.test(e), "detector"), pv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./pieDiagram-NIOCPIFQ-CfWDHy4W.js");
  return { id: gp, diagram: e };
}, "loader"), gv = {
  id: gp,
  detector: dv,
  loader: pv
}, mp = "quadrantChart", mv = /* @__PURE__ */ p((e) => /^\s*quadrantChart/.test(e), "detector"), yv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./quadrantDiagram-2OG54O6I-DQVDfxhj.js");
  return { id: mp, diagram: e };
}, "loader"), xv = {
  id: mp,
  detector: mv,
  loader: yv
}, bv = xv, yp = "xychart", Cv = /* @__PURE__ */ p((e) => /^\s*xychart-beta/.test(e), "detector"), kv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./xychartDiagram-H2YORKM3-D5hBlJy3.js");
  return { id: yp, diagram: e };
}, "loader"), wv = {
  id: yp,
  detector: Cv,
  loader: kv
}, _v = wv, xp = "requirement", vv = /* @__PURE__ */ p((e) => /^\s*requirement(Diagram)?/.test(e), "detector"), Sv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./requirementDiagram-QOLK2EJ7-CdOkbr04.js");
  return { id: xp, diagram: e };
}, "loader"), Tv = {
  id: xp,
  detector: vv,
  loader: Sv
}, Bv = Tv, bp = "sequence", Lv = /* @__PURE__ */ p((e) => /^\s*sequenceDiagram/.test(e), "detector"), Mv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./sequenceDiagram-SKLFT4DO-DZQ3bIau.js");
  return { id: bp, diagram: e };
}, "loader"), $v = {
  id: bp,
  detector: Lv,
  loader: Mv
}, Av = $v, Cp = "class", Fv = /* @__PURE__ */ p((e, t) => {
  var r;
  return ((r = t == null ? void 0 : t.class) == null ? void 0 : r.defaultRenderer) === "dagre-wrapper" ? !1 : /^\s*classDiagram/.test(e);
}, "detector"), Ev = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./classDiagram-M3E45YP4-Db19YdWQ.js");
  return { id: Cp, diagram: e };
}, "loader"), Ov = {
  id: Cp,
  detector: Fv,
  loader: Ev
}, Dv = Ov, kp = "classDiagram", Rv = /* @__PURE__ */ p((e, t) => {
  var r;
  return /^\s*classDiagram/.test(e) && ((r = t == null ? void 0 : t.class) == null ? void 0 : r.defaultRenderer) === "dagre-wrapper" ? !0 : /^\s*classDiagram-v2/.test(e);
}, "detector"), Pv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./classDiagram-v2-YAWTLIQI-Db19YdWQ.js");
  return { id: kp, diagram: e };
}, "loader"), Iv = {
  id: kp,
  detector: Rv,
  loader: Pv
}, Nv = Iv, wp = "state", zv = /* @__PURE__ */ p((e, t) => {
  var r;
  return ((r = t == null ? void 0 : t.state) == null ? void 0 : r.defaultRenderer) === "dagre-wrapper" ? !1 : /^\s*stateDiagram/.test(e);
}, "detector"), qv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./stateDiagram-MI5ZYTHO-BULS-V4S.js");
  return { id: wp, diagram: e };
}, "loader"), Wv = {
  id: wp,
  detector: zv,
  loader: qv
}, Hv = Wv, _p = "stateDiagram", jv = /* @__PURE__ */ p((e, t) => {
  var r;
  return !!(/^\s*stateDiagram-v2/.test(e) || /^\s*stateDiagram/.test(e) && ((r = t == null ? void 0 : t.state) == null ? void 0 : r.defaultRenderer) === "dagre-wrapper");
}, "detector"), Yv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./stateDiagram-v2-5AN5P6BG-Cv1LJJ9I.js");
  return { id: _p, diagram: e };
}, "loader"), Gv = {
  id: _p,
  detector: jv,
  loader: Yv
}, Uv = Gv, vp = "journey", Xv = /* @__PURE__ */ p((e) => /^\s*journey/.test(e), "detector"), Vv = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./journeyDiagram-EWQZEKCU-B-CKpT3H.js");
  return { id: vp, diagram: e };
}, "loader"), Zv = {
  id: vp,
  detector: Xv,
  loader: Vv
}, Kv = Zv, Qv = /* @__PURE__ */ p((e, t, r) => {
  F.debug(`rendering svg for syntax error
`);
  const i = l_(t), a = i.append("g");
  i.attr("viewBox", "0 0 2412 512"), ec(i, 100, 512, !0), a.append("path").attr("class", "error-icon").attr(
    "d",
    "m411.313,123.313c6.25-6.25 6.25-16.375 0-22.625s-16.375-6.25-22.625,0l-32,32-9.375,9.375-20.688-20.688c-12.484-12.5-32.766-12.5-45.25,0l-16,16c-1.261,1.261-2.304,2.648-3.31,4.051-21.739-8.561-45.324-13.426-70.065-13.426-105.867,0-192,86.133-192,192s86.133,192 192,192 192-86.133 192-192c0-24.741-4.864-48.327-13.426-70.065 1.402-1.007 2.79-2.049 4.051-3.31l16-16c12.5-12.492 12.5-32.758 0-45.25l-20.688-20.688 9.375-9.375 32.001-31.999zm-219.313,100.687c-52.938,0-96,43.063-96,96 0,8.836-7.164,16-16,16s-16-7.164-16-16c0-70.578 57.422-128 128-128 8.836,0 16,7.164 16,16s-7.164,16-16,16z"
  ), a.append("path").attr("class", "error-icon").attr(
    "d",
    "m459.02,148.98c-6.25-6.25-16.375-6.25-22.625,0s-6.25,16.375 0,22.625l16,16c3.125,3.125 7.219,4.688 11.313,4.688 4.094,0 8.188-1.563 11.313-4.688 6.25-6.25 6.25-16.375 0-22.625l-16.001-16z"
  ), a.append("path").attr("class", "error-icon").attr(
    "d",
    "m340.395,75.605c3.125,3.125 7.219,4.688 11.313,4.688 4.094,0 8.188-1.563 11.313-4.688 6.25-6.25 6.25-16.375 0-22.625l-16-16c-6.25-6.25-16.375-6.25-22.625,0s-6.25,16.375 0,22.625l15.999,16z"
  ), a.append("path").attr("class", "error-icon").attr(
    "d",
    "m400,64c8.844,0 16-7.164 16-16v-32c0-8.836-7.156-16-16-16-8.844,0-16,7.164-16,16v32c0,8.836 7.156,16 16,16z"
  ), a.append("path").attr("class", "error-icon").attr(
    "d",
    "m496,96.586h-32c-8.844,0-16,7.164-16,16 0,8.836 7.156,16 16,16h32c8.844,0 16-7.164 16-16 0-8.836-7.156-16-16-16z"
  ), a.append("path").attr("class", "error-icon").attr(
    "d",
    "m436.98,75.605c3.125,3.125 7.219,4.688 11.313,4.688 4.094,0 8.188-1.563 11.313-4.688l32-32c6.25-6.25 6.25-16.375 0-22.625s-16.375-6.25-22.625,0l-32,32c-6.251,6.25-6.251,16.375-0.001,22.625z"
  ), a.append("text").attr("class", "error-text").attr("x", 1440).attr("y", 250).attr("font-size", "150px").style("text-anchor", "middle").text("Syntax error in text"), a.append("text").attr("class", "error-text").attr("x", 1250).attr("y", 400).attr("font-size", "100px").style("text-anchor", "middle").text(`mermaid version ${r}`);
}, "draw"), Sp = { draw: Qv }, Jv = Sp, tS = {
  db: {},
  renderer: Sp,
  parser: {
    parse: /* @__PURE__ */ p(() => {
    }, "parse")
  }
}, eS = tS, Tp = "flowchart-elk", rS = /* @__PURE__ */ p((e, t = {}) => {
  var r;
  return (
    // If diagram explicitly states flowchart-elk
    /^\s*flowchart-elk/.test(e) || // If a flowchart/graph diagram has their default renderer set to elk
    /^\s*flowchart|graph/.test(e) && ((r = t == null ? void 0 : t.flowchart) == null ? void 0 : r.defaultRenderer) === "elk" ? (t.layout = "elk", !0) : !1
  );
}, "detector"), iS = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./flowDiagram-KYDEHFYC-B_9iElls.js");
  return { id: Tp, diagram: e };
}, "loader"), aS = {
  id: Tp,
  detector: rS,
  loader: iS
}, nS = aS, Bp = "timeline", sS = /* @__PURE__ */ p((e) => /^\s*timeline/.test(e), "detector"), oS = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./timeline-definition-MYPXXCX6-G2tHPlqJ.js");
  return { id: Bp, diagram: e };
}, "loader"), lS = {
  id: Bp,
  detector: sS,
  loader: oS
}, cS = lS, Lp = "mindmap", hS = /* @__PURE__ */ p((e) => /^\s*mindmap/.test(e), "detector"), uS = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./mindmap-definition-6CBA2TL7-dreKDQK6.js");
  return { id: Lp, diagram: e };
}, "loader"), fS = {
  id: Lp,
  detector: hS,
  loader: uS
}, dS = fS, Mp = "kanban", pS = /* @__PURE__ */ p((e) => /^\s*kanban/.test(e), "detector"), gS = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./kanban-definition-ZSS6B67P-auixNrPN.js");
  return { id: Mp, diagram: e };
}, "loader"), mS = {
  id: Mp,
  detector: pS,
  loader: gS
}, yS = mS, $p = "sankey", xS = /* @__PURE__ */ p((e) => /^\s*sankey-beta/.test(e), "detector"), bS = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./sankeyDiagram-4UZDY2LN-iqWOs7So.js");
  return { id: $p, diagram: e };
}, "loader"), CS = {
  id: $p,
  detector: xS,
  loader: bS
}, kS = CS, Ap = "packet", wS = /* @__PURE__ */ p((e) => /^\s*packet(-beta)?/.test(e), "detector"), _S = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./diagram-5UYTHUR4-BlSoKXS9.js");
  return { id: Ap, diagram: e };
}, "loader"), vS = {
  id: Ap,
  detector: wS,
  loader: _S
}, Fp = "radar", SS = /* @__PURE__ */ p((e) => /^\s*radar-beta/.test(e), "detector"), TS = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./diagram-ZTM2IBQH-Dh5rMqmx.js");
  return { id: Fp, diagram: e };
}, "loader"), BS = {
  id: Fp,
  detector: SS,
  loader: TS
}, Ep = "block", LS = /* @__PURE__ */ p((e) => /^\s*block-beta/.test(e), "detector"), MS = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./blockDiagram-6J76NXCF-zMofcCQe.js");
  return { id: Ep, diagram: e };
}, "loader"), $S = {
  id: Ep,
  detector: LS,
  loader: MS
}, AS = $S, Op = "architecture", FS = /* @__PURE__ */ p((e) => /^\s*architecture/.test(e), "detector"), ES = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./architectureDiagram-SUXI7LT5-1KF4Is23.js");
  return { id: Op, diagram: e };
}, "loader"), OS = {
  id: Op,
  detector: FS,
  loader: ES
}, DS = OS, Dp = "treemap", RS = /* @__PURE__ */ p((e) => /^\s*treemap/.test(e), "detector"), PS = /* @__PURE__ */ p(async () => {
  const { diagram: e } = await import("./diagram-VMROVX33-TttR8MAt.js");
  return { id: Dp, diagram: e };
}, "loader"), IS = {
  id: Dp,
  detector: RS,
  loader: PS
}, Fl = !1, Za = /* @__PURE__ */ p(() => {
  Fl || (Fl = !0, Gi("error", eS, (e) => e.toLowerCase().trim() === "error"), Gi(
    "---",
    // --- diagram type may appear if YAML front-matter is not parsed correctly
    {
      db: {
        clear: /* @__PURE__ */ p(() => {
        }, "clear")
      },
      styles: {},
      // should never be used
      renderer: {
        draw: /* @__PURE__ */ p(() => {
        }, "draw")
      },
      parser: {
        parse: /* @__PURE__ */ p(() => {
          throw new Error(
            "Diagrams beginning with --- are not valid. If you were trying to use a YAML front-matter, please ensure that you've correctly opened and closed the YAML front-matter with un-indented `---` blocks"
          );
        }, "parse")
      },
      init: /* @__PURE__ */ p(() => null, "init")
      // no op
    },
    (e) => e.toLowerCase().trimStart().startsWith("---")
  ), gn(nS, dS, DS), gn(
    H_,
    yS,
    Nv,
    Dv,
    ev,
    cv,
    fv,
    gv,
    Bv,
    Av,
    K_,
    U_,
    cS,
    nv,
    Uv,
    Hv,
    Kv,
    bv,
    kS,
    vS,
    _v,
    AS,
    BS,
    IS
  ));
}, "addDiagrams"), NS = /* @__PURE__ */ p(async () => {
  F.debug("Loading registered diagrams");
  const t = (await Promise.allSettled(
    Object.entries(ze).map(async ([r, { detector: i, loader: a }]) => {
      if (a)
        try {
          bn(r);
        } catch {
          try {
            const { diagram: n, id: o } = await a();
            Gi(o, n, i);
          } catch (n) {
            throw F.error(`Failed to load external diagram with key ${r}. Removing from detectors.`), delete ze[r], n;
          }
        }
    })
  )).filter((r) => r.status === "rejected");
  if (t.length > 0) {
    F.error(`Failed to load ${t.length} external diagrams`);
    for (const r of t)
      F.error(r);
    throw new Error(`Failed to load ${t.length} external diagrams`);
  }
}, "loadRegisteredDiagrams"), zS = "graphics-document document";
function Rp(e, t) {
  e.attr("role", zS), t !== "" && e.attr("aria-roledescription", t);
}
p(Rp, "setA11yDiagramInfo");
function Pp(e, t, r, i) {
  if (e.insert !== void 0) {
    if (r) {
      const a = `chart-desc-${i}`;
      e.attr("aria-describedby", a), e.insert("desc", ":first-child").attr("id", a).text(r);
    }
    if (t) {
      const a = `chart-title-${i}`;
      e.attr("aria-labelledby", a), e.insert("title", ":first-child").attr("id", a).text(t);
    }
  }
}
p(Pp, "addSVGa11yTitleDescription");
var Ne, ls = (Ne = class {
  constructor(t, r, i, a, n) {
    this.type = t, this.text = r, this.db = i, this.parser = a, this.renderer = n;
  }
  static async fromText(t, r = {}) {
    var c, h;
    const i = It(), a = us(t, i);
    t = PC(t) + `
`;
    try {
      bn(a);
    } catch {
      const u = hg(a);
      if (!u)
        throw new ql(`Diagram ${a} not found.`);
      const { id: f, diagram: d } = await u();
      Gi(f, d);
    }
    const { db: n, parser: o, renderer: s, init: l } = bn(a);
    return o.parser && (o.parser.yy = n), (c = n.clear) == null || c.call(n), l == null || l(i), r.title && ((h = n.setDiagramTitle) == null || h.call(n, r.title)), await o.parse(t), new Ne(a, t, n, o, s);
  }
  async render(t, r) {
    await this.renderer.draw(this.text, t, r, this);
  }
  getParser() {
    return this.parser;
  }
  getType() {
    return this.type;
  }
}, p(Ne, "Diagram"), Ne), El = [], qS = /* @__PURE__ */ p(() => {
  El.forEach((e) => {
    e();
  }), El = [];
}, "attachFunctions"), WS = /* @__PURE__ */ p((e) => e.replace(/^\s*%%(?!{)[^\n]+\n?/gm, "").trimStart(), "cleanupComments");
function Ip(e) {
  const t = e.match(zl);
  if (!t)
    return {
      text: e,
      metadata: {}
    };
  let r = lm(t[1], {
    // To support config, we need JSON schema.
    // https://www.yaml.org/spec/1.2/spec.html#id2803231
    schema: om
  }) ?? {};
  r = typeof r == "object" && !Array.isArray(r) ? r : {};
  const i = {};
  return r.displayMode && (i.displayMode = r.displayMode.toString()), r.title && (i.title = r.title.toString()), r.config && (i.config = r.config), {
    text: e.slice(t[0].length),
    metadata: i
  };
}
p(Ip, "extractFrontMatter");
var HS = /* @__PURE__ */ p((e) => e.replace(/\r\n?/g, `
`).replace(
  /<(\w+)([^>]*)>/g,
  (t, r, i) => "<" + r + i.replace(/="([^"]*)"/g, "='$1'") + ">"
), "cleanupText"), jS = /* @__PURE__ */ p((e) => {
  const { text: t, metadata: r } = Ip(e), { displayMode: i, title: a, config: n = {} } = r;
  return i && (n.gantt || (n.gantt = {}), n.gantt.displayMode = i), { title: a, config: n, text: t };
}, "processFrontmatter"), YS = /* @__PURE__ */ p((e) => {
  const t = Jt.detectInit(e) ?? {}, r = Jt.detectDirective(e, "wrap");
  return Array.isArray(r) ? t.wrap = r.some(({ type: i }) => i === "wrap") : (r == null ? void 0 : r.type) === "wrap" && (t.wrap = !0), {
    text: vC(e),
    directive: t
  };
}, "processDirectives");
function ho(e) {
  const t = HS(e), r = jS(t), i = YS(r.text), a = Us(r.config, i.directive);
  return e = WS(i.text), {
    code: e,
    title: r.title,
    config: a
  };
}
p(ho, "preprocessDiagram");
function Np(e) {
  const t = new TextEncoder().encode(e), r = Array.from(t, (i) => String.fromCodePoint(i)).join("");
  return btoa(r);
}
p(Np, "toBase64");
var GS = 5e4, US = "graph TB;a[Maximum text size in diagram exceeded];style a fill:#faa", XS = "sandbox", VS = "loose", ZS = "http://www.w3.org/2000/svg", KS = "http://www.w3.org/1999/xlink", QS = "http://www.w3.org/1999/xhtml", JS = "100%", tT = "100%", eT = "border:0;margin:0;", rT = "margin:0", iT = "allow-top-navigation-by-user-activation allow-popups", aT = 'The "iframe" tag is not supported by your browser.', nT = ["foreignobject"], sT = ["dominant-baseline"];
function uo(e) {
  const t = ho(e);
  return ji(), Tg(t.config ?? {}), t;
}
p(uo, "processAndSetConfigs");
async function zp(e, t) {
  Za();
  try {
    const { code: r, config: i } = uo(e);
    return { diagramType: (await Wp(r)).type, config: i };
  } catch (r) {
    if (t != null && t.suppressErrors)
      return !1;
    throw r;
  }
}
p(zp, "parse");
var Ol = /* @__PURE__ */ p((e, t, r = []) => `
.${e} ${t} { ${r.join(" !important; ")} !important; }`, "cssImportantStyles"), oT = /* @__PURE__ */ p((e, t = /* @__PURE__ */ new Map()) => {
  var i;
  let r = "";
  if (e.themeCSS !== void 0 && (r += `
${e.themeCSS}`), e.fontFamily !== void 0 && (r += `
:root { --mermaid-font-family: ${e.fontFamily}}`), e.altFontFamily !== void 0 && (r += `
:root { --mermaid-alt-font-family: ${e.altFontFamily}}`), t instanceof Map) {
    const s = e.htmlLabels ?? ((i = e.flowchart) == null ? void 0 : i.htmlLabels) ? ["> *", "span"] : ["rect", "polygon", "ellipse", "circle", "path"];
    t.forEach((l) => {
      Al(l.styles) || s.forEach((c) => {
        r += Ol(l.id, c, l.styles);
      }), Al(l.textStyles) || (r += Ol(
        l.id,
        "tspan",
        ((l == null ? void 0 : l.textStyles) || []).map((c) => c.replace("color", "fill"))
      ));
    });
  }
  return r;
}, "createCssStyles"), lT = /* @__PURE__ */ p((e, t, r, i) => {
  const a = oT(e, r), n = jg(t, a, e.themeVariables);
  return is(__(`${i}{${n}}`), S_);
}, "createUserStyles"), cT = /* @__PURE__ */ p((e = "", t, r) => {
  let i = e;
  return !r && !t && (i = i.replace(
    /marker-end="url\([\d+./:=?A-Za-z-]*?#/g,
    'marker-end="url(#'
  )), i = Ze(i), i = i.replace(/<br>/g, "<br/>"), i;
}, "cleanUpSvgCode"), hT = /* @__PURE__ */ p((e = "", t) => {
  var a, n;
  const r = (n = (a = t == null ? void 0 : t.viewBox) == null ? void 0 : a.baseVal) != null && n.height ? t.viewBox.baseVal.height + "px" : tT, i = Np(`<body style="${rT}">${e}</body>`);
  return `<iframe style="width:${JS};height:${r};${eT}" src="data:text/html;charset=UTF-8;base64,${i}" sandbox="${iT}">
  ${aT}
</iframe>`;
}, "putIntoIFrame"), Dl = /* @__PURE__ */ p((e, t, r, i, a) => {
  const n = e.append("div");
  n.attr("id", r), i && n.attr("style", i);
  const o = n.append("svg").attr("id", t).attr("width", "100%").attr("xmlns", ZS);
  return a && o.attr("xmlns:xlink", a), o.append("g"), e;
}, "appendDivSvgG");
function cs(e, t) {
  return e.append("iframe").attr("id", t).attr("style", "width: 100%; height: 100%;").attr("sandbox", "");
}
p(cs, "sandboxedIframe");
var uT = /* @__PURE__ */ p((e, t, r, i) => {
  var a, n, o;
  (a = e.getElementById(t)) == null || a.remove(), (n = e.getElementById(r)) == null || n.remove(), (o = e.getElementById(i)) == null || o.remove();
}, "removeExistingElements"), fT = /* @__PURE__ */ p(async function(e, t, r) {
  var I, D, B, M, T, A;
  Za();
  const i = uo(t);
  t = i.code;
  const a = It();
  F.debug(a), t.length > ((a == null ? void 0 : a.maxTextSize) ?? GS) && (t = US);
  const n = "#" + e, o = "i" + e, s = "#" + o, l = "d" + e, c = "#" + l, h = /* @__PURE__ */ p(() => {
    const N = et(f ? s : c).node();
    N && "remove" in N && N.remove();
  }, "removeTempElements");
  let u = et("body");
  const f = a.securityLevel === XS, d = a.securityLevel === VS, g = a.fontFamily;
  if (r !== void 0) {
    if (r && (r.innerHTML = ""), f) {
      const L = cs(et(r), o);
      u = et(L.nodes()[0].contentDocument.body), u.node().style.margin = 0;
    } else
      u = et(r);
    Dl(u, e, l, `font-family: ${g}`, KS);
  } else {
    if (uT(document, e, l, o), f) {
      const L = cs(et("body"), o);
      u = et(L.nodes()[0].contentDocument.body), u.node().style.margin = 0;
    } else
      u = et("body");
    Dl(u, e, l);
  }
  let m, y;
  try {
    m = await ls.fromText(t, { title: i.title });
  } catch (L) {
    if (a.suppressErrorRendering)
      throw h(), L;
    m = await ls.fromText("error"), y = L;
  }
  const x = u.select(c).node(), b = m.type, k = x.firstChild, S = k.firstChild, w = (D = (I = m.renderer).getClasses) == null ? void 0 : D.call(I, t, m), C = lT(a, b, w, n), _ = document.createElement("style");
  _.innerHTML = C, k.insertBefore(_, S);
  try {
    await m.renderer.draw(t, e, _l.version, m);
  } catch (L) {
    throw a.suppressErrorRendering ? h() : Jv.draw(t, e, _l.version), L;
  }
  const E = u.select(`${c} svg`), R = (M = (B = m.db).getAccTitle) == null ? void 0 : M.call(B), O = (A = (T = m.db).getAccDescription) == null ? void 0 : A.call(T);
  Hp(b, E, R, O), u.select(`[id="${e}"]`).selectAll("foreignobject > *").attr("xmlns", QS);
  let $ = u.select(c).node().innerHTML;
  if (F.debug("config.arrowMarkerAbsolute", a.arrowMarkerAbsolute), $ = cT($, f, bt(a.arrowMarkerAbsolute)), f) {
    const L = u.select(c + " svg").node();
    $ = hT($, L);
  } else d || ($ = pr.sanitize($, {
    ADD_TAGS: nT,
    ADD_ATTR: sT,
    HTML_INTEGRATION_POINTS: { foreignobject: !0 }
  }));
  if (qS(), y)
    throw y;
  return h(), {
    diagramType: b,
    svg: $,
    bindFunctions: m.db.bindFunctions
  };
}, "render");
function qp(e = {}) {
  var i;
  const t = vt({}, e);
  t != null && t.fontFamily && !((i = t.themeVariables) != null && i.fontFamily) && (t.themeVariables || (t.themeVariables = {}), t.themeVariables.fontFamily = t.fontFamily), vg(t), t != null && t.theme && t.theme in ue ? t.themeVariables = ue[t.theme].getThemeVariables(
    t.themeVariables
  ) : t && (t.themeVariables = ue.default.getThemeVariables(t.themeVariables));
  const r = typeof t == "object" ? _g(t) : Ul();
  hs(r.logLevel), Za();
}
p(qp, "initialize");
var Wp = /* @__PURE__ */ p((e, t = {}) => {
  const { code: r } = ho(e);
  return ls.fromText(r, t);
}, "getDiagramFromText");
function Hp(e, t, r, i) {
  Rp(t, e), Pp(t, r, i, t.attr("id"));
}
p(Hp, "addA11yInfo");
var Ge = Object.freeze({
  render: fT,
  parse: zp,
  getDiagramFromText: Wp,
  initialize: qp,
  getConfig: It,
  setConfig: Xl,
  getSiteConfig: Ul,
  updateSiteConfig: Sg,
  reset: /* @__PURE__ */ p(() => {
    ji();
  }, "reset"),
  globalReset: /* @__PURE__ */ p(() => {
    ji(gr);
  }, "globalReset"),
  defaultConfig: gr
});
hs(It().logLevel);
ji(It());
var dT = /* @__PURE__ */ p((e, t, r) => {
  F.warn(e), Gs(e) ? (r && r(e.str, e.hash), t.push({ ...e, message: e.str, error: e })) : (r && r(e), e instanceof Error && t.push({
    str: e.message,
    message: e.message,
    hash: e.name,
    error: e
  }));
}, "handleError"), jp = /* @__PURE__ */ p(async function(e = {
  querySelector: ".mermaid"
}) {
  try {
    await pT(e);
  } catch (t) {
    if (Gs(t) && F.error(t.str), zt.parseError && zt.parseError(t), !e.suppressErrors)
      throw F.error("Use the suppressErrors option to suppress these errors"), t;
  }
}, "run"), pT = /* @__PURE__ */ p(async function({ postRenderCallback: e, querySelector: t, nodes: r } = {
  querySelector: ".mermaid"
}) {
  const i = Ge.getConfig();
  F.debug(`${e ? "" : "No "}Callback function found`);
  let a;
  if (r)
    a = r;
  else if (t)
    a = document.querySelectorAll(t);
  else
    throw new Error("Nodes and querySelector are both undefined");
  F.debug(`Found ${a.length} diagrams`), (i == null ? void 0 : i.startOnLoad) !== void 0 && (F.debug("Start On Load: " + (i == null ? void 0 : i.startOnLoad)), Ge.updateSiteConfig({ startOnLoad: i == null ? void 0 : i.startOnLoad }));
  const n = new Jt.InitIDGenerator(i.deterministicIds, i.deterministicIDSeed);
  let o;
  const s = [];
  for (const l of Array.from(a)) {
    if (F.info("Rendering diagram: " + l.id), l.getAttribute("data-processed"))
      continue;
    l.setAttribute("data-processed", "true");
    const c = `mermaid-${n.next()}`;
    o = l.innerHTML, o = mf(Jt.entityDecode(o)).trim().replace(/<br\s*\/?>/gi, "<br/>");
    const h = Jt.detectInit(o);
    h && F.debug("Detected early reinit: ", h);
    try {
      const { svg: u, bindFunctions: f } = await Xp(c, o, l);
      l.innerHTML = u, e && await e(c), f && f(l);
    } catch (u) {
      dT(u, s, zt.parseError);
    }
  }
  if (s.length > 0)
    throw s[0];
}, "runThrowsErrors"), Yp = /* @__PURE__ */ p(function(e) {
  Ge.initialize(e);
}, "initialize"), gT = /* @__PURE__ */ p(async function(e, t, r) {
  F.warn("mermaid.init is deprecated. Please use run instead."), e && Yp(e);
  const i = { postRenderCallback: r, querySelector: ".mermaid" };
  typeof t == "string" ? i.querySelector = t : t && (t instanceof HTMLElement ? i.nodes = [t] : i.nodes = t), await jp(i);
}, "init"), mT = /* @__PURE__ */ p(async (e, {
  lazyLoad: t = !0
} = {}) => {
  Za(), gn(...e), t === !1 && await NS();
}, "registerExternalDiagrams"), Gp = /* @__PURE__ */ p(function() {
  if (zt.startOnLoad) {
    const { startOnLoad: e } = Ge.getConfig();
    e && zt.run().catch((t) => F.error("Mermaid failed to initialize", t));
  }
}, "contentLoaded");
typeof document < "u" && window.addEventListener("load", Gp, !1);
var yT = /* @__PURE__ */ p(function(e) {
  zt.parseError = e;
}, "setParseErrorHandler"), _a = [], dn = !1, Up = /* @__PURE__ */ p(async () => {
  if (!dn) {
    for (dn = !0; _a.length > 0; ) {
      const e = _a.shift();
      if (e)
        try {
          await e();
        } catch (t) {
          F.error("Error executing queue", t);
        }
    }
    dn = !1;
  }
}, "executeQueue"), xT = /* @__PURE__ */ p(async (e, t) => new Promise((r, i) => {
  const a = /* @__PURE__ */ p(() => new Promise((n, o) => {
    Ge.parse(e, t).then(
      (s) => {
        n(s), r(s);
      },
      (s) => {
        var l;
        F.error("Error parsing", s), (l = zt.parseError) == null || l.call(zt, s), o(s), i(s);
      }
    );
  }), "performCall");
  _a.push(a), Up().catch(i);
}), "parse"), Xp = /* @__PURE__ */ p((e, t, r) => new Promise((i, a) => {
  const n = /* @__PURE__ */ p(() => new Promise((o, s) => {
    Ge.render(e, t, r).then(
      (l) => {
        o(l), i(l);
      },
      (l) => {
        var c;
        F.error("Error parsing", l), (c = zt.parseError) == null || c.call(zt, l), s(l), a(l);
      }
    );
  }), "performCall");
  _a.push(n), Up().catch(a);
}), "render"), bT = /* @__PURE__ */ p(() => Object.keys(ze).map((e) => ({
  id: e
})), "getRegisteredDiagramsMetadata"), zt = {
  startOnLoad: !0,
  mermaidAPI: Ge,
  parse: xT,
  render: Xp,
  init: gT,
  run: jp,
  registerExternalDiagrams: mT,
  registerLayoutLoaders: ep,
  initialize: Yp,
  parseError: void 0,
  contentLoaded: Gp,
  setParseErrorHandler: yT,
  detectType: us,
  registerIconPacks: qk,
  getRegisteredDiagramsMetadata: bT
}, CT = zt;
/*! Check if previously processed */
/*!
 * Wait for document loaded before starting the execution
 */
const ZT = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: CT
}, Symbol.toStringTag, { value: "Module" }));
export {
  rc as $,
  Gr as A,
  lg as B,
  Jg as C,
  Us as D,
  It as E,
  Yl as F,
  MC as G,
  l_ as H,
  _l as I,
  om as J,
  mg as K,
  yr as L,
  _T as M,
  qa as N,
  tc as O,
  fs as P,
  go as Q,
  Pb as R,
  Ri as S,
  LC as T,
  hi as U,
  qg as V,
  ci as W,
  z as X,
  X as Y,
  CC as Z,
  p as _,
  Ug as a,
  sx as a$,
  Eb as a0,
  Es as a1,
  MT as a2,
  FT as a3,
  Qe as a4,
  Io as a5,
  Po as a6,
  OT as a7,
  ET as a8,
  AT as a9,
  mC as aA,
  l2 as aB,
  cC as aC,
  Is as aD,
  Al as aE,
  Hk as aF,
  Ob as aG,
  yi as aH,
  qk as aI,
  zk as aJ,
  Ms as aK,
  Ce as aL,
  Ao as aM,
  dx as aN,
  ii as aO,
  Ue as aP,
  yC as aQ,
  Gu as aR,
  Ra as aS,
  za as aT,
  fa as aU,
  Xu as aV,
  Yu as aW,
  U2 as aX,
  Y as aY,
  Ih as aZ,
  Pt as a_,
  BT as aa,
  LT as ab,
  RT as ac,
  $T as ad,
  DT as ae,
  pw as af,
  Jd as ag,
  GT as ah,
  cm as ai,
  bt as aj,
  Le as ak,
  vs as al,
  Lf as am,
  Ze as an,
  tf as ao,
  K as ap,
  ee as aq,
  n_ as ar,
  YT as as,
  UT as at,
  HT as au,
  j as av,
  jT as aw,
  qw as ax,
  Pw as ay,
  Rw as az,
  Gg as b,
  Ls as b0,
  Jh as b1,
  di as b2,
  ru as b3,
  TT as b4,
  tg as b5,
  gC as b6,
  lC as b7,
  V1 as b8,
  Ns as b9,
  Na as bA,
  ZT as bB,
  H2 as ba,
  bC as bb,
  gi as bc,
  Tr as bd,
  ca as be,
  tC as bf,
  M_ as bg,
  pi as bh,
  ua as bi,
  X2 as bj,
  Nu as bk,
  Q1 as bl,
  J1 as bm,
  Fe as bn,
  Jo as bo,
  t2 as bp,
  zs as bq,
  K1 as br,
  i2 as bs,
  Br as bt,
  Be as bu,
  Xo as bv,
  qs as bw,
  qu as bx,
  ss as by,
  xC as bz,
  at as c,
  et as d,
  ec as e,
  vt as f,
  Vg as g,
  pe as h,
  qe as i,
  Nh as j,
  vr as k,
  F as l,
  rf as m,
  vT as n,
  VT as o,
  Zg as p,
  Kg as q,
  XT as r,
  Xg as s,
  lm as t,
  Jt as u,
  Ew as v,
  FC as w,
  PT as x,
  Yg as y,
  ST as z
};
