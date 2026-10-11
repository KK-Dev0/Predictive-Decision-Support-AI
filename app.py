(() => {
  "use strict";
  const cfg = JSON.parse(document.getElementById("config").textContent);
  const form = document.getElementById("assess-form");
  const btn = document.getElementById("run");
  const errBox = document.getElementById("form-error");
  const result = document.getElementById("result");
  const empty = document.getElementById("result-empty");
  const body = document.getElementById("result-body");
  const BAND_TEXT = { approve: "Likely approve", review: "Manual review recommended", decline: "Likely decline" };

  // Example borrowers (inputs only, the model decides what they mean)
  const PRESETS = {
    steady: { age: 52, MonthlyIncome: 6500, NumberOfDependents: 2, RevolvingUtilizationOfUnsecuredLines: 12, DebtRatio: 25,
              NumberOfOpenCreditLinesAndLoans: 9, NumberRealEstateLoansOrLines: 1, "NumberOfTime30-59DaysPastDueNotWorse": 0,
              "NumberOfTime60-89DaysPastDueNotWorse": 0, NumberOfTimes90DaysLate: 0 },
    heavy: { age: 41, MonthlyIncome: 4200, NumberOfDependents: 3, RevolvingUtilizationOfUnsecuredLines: 85, DebtRatio: 55,
             NumberOfOpenCreditLinesAndLoans: 12, NumberRealEstateLoansOrLines: 2, "NumberOfTime30-59DaysPastDueNotWorse": 1,
             "NumberOfTime60-89DaysPastDueNotWorse": 0, NumberOfTimes90DaysLate: 0 },
    late: { age: 29, MonthlyIncome: 3000, NumberOfDependents: 1, RevolvingUtilizationOfUnsecuredLines: 95, DebtRatio: 70,
            NumberOfOpenCreditLinesAndLoans: 6, NumberRealEstateLoansOrLines: 0, "NumberOfTime30-59DaysPastDueNotWorse": 2,
            "NumberOfTime60-89DaysPastDueNotWorse": 1, NumberOfTimes90DaysLate: 1 },
  };

  document.querySelectorAll("[data-unknown]").forEach((cb) => {
    const input = form.elements[cb.dataset.unknown];
    cb.addEventListener("change", () => { input.disabled = cb.checked; });
  });

  document.querySelectorAll("[data-preset]").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("[data-unknown]").forEach((cb) => { cb.checked = false; });
      for (const [k, v] of Object.entries(PRESETS[b.dataset.preset])) {
        const el = form.elements[k];
        if (el) { el.disabled = false; el.value = v; }
      }
      errBox.hidden = true;
    });
  });

  // Zone bar and tick labels come from the model's thresholds
  const axis = cfg.axis_max;
  const zones = document.querySelectorAll("#zonebar .zone");
  zones[0].style.flex = `0 0 ${(cfg.t_review / axis) * 100}%`;
  zones[1].style.flex = `0 0 ${((cfg.t_star - cfg.t_review) / axis) * 100}%`;
  zones[2].style.flex = "1 1 0";
  const ticks = document.getElementById("ticks");
  const pc = (v) => `${+(v * 100).toFixed(1)}%`;
  [[0, "0%"], [cfg.t_review, pc(cfg.t_review)], [cfg.t_star, pc(cfg.t_star)], [axis, `${pc(axis)}+`]]
    .forEach(([v, text]) => {
      const s = document.createElement("span");
      s.style.left = `${(v / axis) * 100}%`;
      s.textContent = text;
      ticks.appendChild(s);
    });

  const legend = document.getElementById("legend");
  [["a", `Approve below ${pc(cfg.t_review)}`], ["r", `Review ${pc(cfg.t_review)} to ${pc(cfg.t_star)}`], ["d", `Decline from ${pc(cfg.t_star)}`]]
    .forEach(([cls, text]) => {
      const s = document.createElement("span");
      const dot = document.createElement("span");
      dot.className = `dot ${cls}`;
      s.append(dot, document.createTextNode(text));
      legend.appendChild(s);
    });

  function payload() {
    const out = {};
    form.querySelectorAll("input[type=number]").forEach((el) => {
      if (el.disabled) { out[el.name] = null; return; }
      const v = parseFloat(el.value);
      out[el.name] = el.dataset.kind === "percent" ? v / 100 : v;
    });
    return out;
  }

  function fillFactors(id, items, cls) {
    const ul = document.getElementById(id);
    ul.className = `flist ${cls}`;
    ul.replaceChildren();
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "none";
      li.textContent = "No single input stands out.";
      ul.appendChild(li);
      return;
    }
    const max = Math.max(...items.map((i) => Math.abs(i.points)));
    items.forEach((i) => {
      const li = document.createElement("li");
      const row = document.createElement("span");
      row.className = "row";
      const name = document.createElement("span");
      name.textContent = i.label;
      if (i.sensitive) {
        const s = document.createElement("span");
        s.className = "sens";
        s.textContent = " (sensitive)";
        name.appendChild(s);
      }
      const val = document.createElement("span");
      val.className = "val";
      val.textContent = `${i.points > 0 ? "+" : "-"}${Math.abs(i.points).toFixed(1)} pts`;
      row.append(name, val);
      const track = document.createElement("span");
      track.className = "track";
      const fill = document.createElement("span");
      fill.className = "fill";
      fill.style.width = `${(Math.abs(i.points) / max) * 100}%`;
      track.appendChild(fill);
      li.append(row, track);
      ul.appendChild(li);
    });
  }

  function render(d) {
    const p = d.probability;
    empty.hidden = true;
    body.hidden = false;
    document.getElementById("prob").textContent = (p * 100).toFixed(1);
    const verdict = document.getElementById("verdict");
    verdict.textContent = BAND_TEXT[d.band];
    verdict.dataset.band = d.band;
    document.getElementById("marker").style.left = `${Math.min(p / axis, 1) * 100}%`;
    const ratio = p / d.base_rate;
    const rel = ratio >= 1.1 ? `about ${ratio.toFixed(1)} times the average` : ratio <= 0.9 ? `about ${(1 / ratio).toFixed(1)} times lower than the average` : "close to the average";
    document.getElementById("context").textContent =
      `The average borrower in the training data has a ${(d.base_rate * 100).toFixed(1)}% chance, so this estimate is ${rel}.`;
    fillFactors("up", d.raising, "up");
    fillFactors("down", d.lowering, "down");
    if (window.matchMedia("(max-width: 920px)").matches) result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errBox.hidden = true;
    if (!form.reportValidity()) return;
    btn.disabled = true;
    btn.textContent = "Assessing...";
    try {
      const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong. Please try again.");
      render(data);
    } catch (ex) {
      errBox.textContent = ex.message;
      errBox.hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "Assess borrower";
    }
  });
})();
