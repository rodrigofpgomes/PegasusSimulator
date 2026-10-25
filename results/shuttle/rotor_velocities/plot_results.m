%% Shuttle - Direct Rotor Velocity Control: figures of the section
% Data: data/*.csv written by export_results.py. All text uses the LaTeX interpreter, i.e. Computer Modern.
% Same figures as plot_results.py (matplotlib), drawn with MATLAB; both write figures/ (PDF + PNG).
%
%   plot_results             % all figures, saved to figures/ (PDF + PNG)
%   plot_results(false)      % draw only, do not save
%
% Figures, in the order of the text:
%   fig01 SAC training                     fig02 PPO training (initial, tanh, tanh + sigma_min)
%   fig03 final policies, one marker per seed               fig04 example episode (rotor commands and error)
%   fig05 reward and observation variants  fig06 figure-eight tracking (RMSE and lag)  fig07 figure-eight paths
% Requires MATLAB R2020b or later (tiledlayout, exportgraphics).

function plot_results(saveFigures)
if nargin < 1, saveFigures = true; end
here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, 'data');
figDir  = fullfile(here, 'figures');
if saveFigures && ~exist(figDir, 'dir'), mkdir(figDir); end

% Computer Modern: every text object through the LaTeX interpreter
names = {'defaultTextInterpreter', 'defaultAxesTickLabelInterpreter', 'defaultLegendInterpreter', ...
         'defaultColorbarTickLabelInterpreter', 'defaultAxesFontName', 'defaultTextFontName', ...
         'defaultLegendFontName', 'defaultAxesFontSize', 'defaultTextFontSize', 'defaultLegendFontSize'};
previous = cellfun(@(n) get(groot, n), names, 'UniformOutput', false);
for k = 1:4, set(groot, names{k}, 'latex'); end      % the first four are interpreters
set(groot, 'defaultAxesFontName', 'Computer Modern', 'defaultTextFontName', 'Computer Modern', ...
           'defaultLegendFontName', 'Computer Modern', 'defaultAxesFontSize', 9, 'defaultTextFontSize', 9, ...
           'defaultLegendFontSize', 9);
restore = onCleanup(@() restoreDefaults(names, previous));

rd = @(f) readtable(fullfile(dataDir, f), 'TextType', 'string', 'TreatAsMissing', {'nan', 'NaN'});
T = rd('training_curves.csv');
E = rd('eval_checkpoints.csv');
M = rd('run_metrics.csv');
X = rd('example_episode.csv');
S = style();

fig1SacTraining(S, T, E, figDir, saveFigures);
fig2PpoTraining(S, T, E, M, figDir, saveFigures);
fig3FinalPerSeed(S, M, figDir, saveFigures);
fig4ExampleEpisode(S, X, figDir, saveFigures);
if isfile(fullfile(dataDir, 'ablation_metrics.csv'))
    fig5Ablation(S, rd('ablation_metrics.csv'), figDir, saveFigures);
end
if isfile(fullfile(dataDir, 'lemniscate_metrics.csv'))
    fig6Lemniscate(S, rd('lemniscate_metrics.csv'), figDir, saveFigures);
end
if isfile(fullfile(dataDir, 'lemniscate_paths.csv'))
    fig7LemniscatePaths(S, rd('lemniscate_paths.csv'), figDir, saveFigures);
end

%% console summary (final policies)
disp(groupsummary(M, 'variant', 'mean', {'survival', 'err_final_cm', 'err_rms_cm', 'sat_pct', 'd_action', 'w_rms'}));
end

%% =============================================================== figures

function fig1SacTraining(S, T, E, figDir, doSave)
% SAC: position error and survival against the transitions (training and deterministic evaluation).
Ts = sortrows(T(T.variant == "sac", :), 'transitions');
Es = sortrows(E(E.variant == "sac", :), 'transitions');
x = Ts.transitions / 1e6;
xe = Es.transitions / 1e6;
[f, tl] = twoPanels(S);

ax = nexttile(tl); hold(ax, 'on');
h = gobjects(1, 3);
h(1) = plot(ax, x, Ts.final_distance_m, '.', 'Color', S.light, 'MarkerSize', 5);
h(2) = plot(ax, x, movmedian(Ts.final_distance_m, 25, 'omitnan'), '-', 'Color', S.sac, 'LineWidth', 1.1);
h(3) = plot(ax, xe, Es.err_final_cm / 100, '-o', 'Color', S.green, 'LineWidth', 1.3, 'MarkerSize', 4, ...
    'MarkerFaceColor', S.green);
set(ax, 'YScale', 'log'); ylim(ax, [0.02 2]); xlim(ax, [0 1]);
xlabel(ax, 'Transitions [$\times 10^{6}$]'); ylabel(ax, 'Position error [m]');
title(ax, '(a) Position error', 'FontWeight', 'normal');
formatAxis(ax, S);
legend(ax, h, {'Training episodes (end of episode)', 'Training, moving median', 'Evaluation, last 2\,s'}, ...
    'Location', 'northeast', 'Box', 'off', 'Interpreter', 'latex');

ax = nexttile(tl); hold(ax, 'on');
h = gobjects(1, 2);
h(1) = plot(ax, x, movmean(1 - Ts.died, 25, 'omitnan'), '-', 'Color', S.sac, 'LineWidth', 1.1);
h(2) = plot(ax, xe, Es.survival, '-o', 'Color', S.green, 'LineWidth', 1.3, 'MarkerSize', 4, 'MarkerFaceColor', S.green);
ylim(ax, [0 1.02]); xlim(ax, [0 1]);
xlabel(ax, 'Transitions [$\times 10^{6}$]'); ylabel(ax, 'Survival rate');
title(ax, '(b) Survival', 'FontWeight', 'normal');
formatAxis(ax, S);
legend(ax, h, {'Training, moving mean', 'Evaluation'}, 'Location', 'southeast', 'Box', 'off', 'Interpreter', 'latex');
saveFigure(f, figDir, 'fig01_sac_training', doSave);
end

function fig2PpoTraining(S, T, E, M, figDir, doSave)
% PPO variants against the iteration; lines are the mean and bands the range over the seeds.
variants = ["initial", "tanh", "tanh_std"];
labels = {'PPO, initial', 'PPO, $\tanh$', 'PPO, $\tanh$ + $\sigma_{\min}$', 'SAC, final ($10^{6}$ transitions)'};
panels = {'final_distance_m', 'Position error [m]',          '(a) Training, end of episode',       'T'; ...
          'err_final_cm',     'Position error [cm]',          '(b) Evaluation, last 2\,s',          'E'; ...
          'sat_pct',          'Commands at $\pm 1$ [\%]',     '(c) Evaluation, saturated commands', 'E'; ...
          'survival',         'Survival rate',                '(d) Evaluation, survival',           'E'};
sac = M(M.variant == "sac", :);
f = newFigure(S, 16, 11.5);
tl = tiledlayout(f, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
h = gobjects(1, 4);
for p = 1:4
    ax = nexttile(tl); hold(ax, 'on');
    isTraining = panels{p, 4} == 'T';
    if isTraining, D = T; else, D = E; end
    for v = 1:3
        [x, Y] = bySeed(D(D.variant == variants(v), :), 'iteration', panels{p, 1});
        if isTraining
            for j = 1:size(Y, 2), Y(:, j) = movmean(Y(:, j), 5, 'omitnan'); end
        end
        lo = min(Y, [], 2, 'omitnan'); hi = max(Y, [], 2, 'omitnan'); mu = mean(Y, 2, 'omitnan');
        ok = ~isnan(lo) & ~isnan(hi);
        fill(ax, [x(ok); flipud(x(ok))], [lo(ok); flipud(hi(ok))], S.variant(v, :), 'FaceAlpha', 0.18, ...
            'EdgeColor', 'none', 'HandleVisibility', 'off');
        if isTraining
            hh = plot(ax, x, mu, '-', 'Color', S.variant(v, :), 'LineWidth', 1.3);
        else
            hh = plot(ax, x, mu, '-o', 'Color', S.variant(v, :), 'LineWidth', 1.3, 'MarkerSize', 3, ...
                'MarkerFaceColor', S.variant(v, :));
        end
        if p == 2, h(v) = hh; end
    end
    if ~isTraining
        hs = plot(ax, [0 600], sac.(panels{p, 1})(1) * [1 1], '--', 'Color', S.sac, 'LineWidth', 1.0);
        if p == 2, h(4) = hs; end
    end
    xlim(ax, [0 600]); xlabel(ax, 'PPO iteration'); ylabel(ax, panels{p, 2});
    title(ax, panels{p, 3}, 'FontWeight', 'normal');
    formatAxis(ax, S);
    switch p
        case 1, set(ax, 'YScale', 'log'); ylim(ax, [0.05 2]);
        case 2, set(ax, 'YScale', 'log'); ylim(ax, [3 150]);
        case 3, ylim(ax, [0 100]);
        case 4, ylim(ax, [0 1.02]);
    end
end
lgd = legend(h, labels, 'Orientation', 'horizontal', 'Box', 'off', 'Interpreter', 'latex');
lgd.Layout.Tile = 'north';
saveFigure(f, figDir, 'fig02_ppo2_training', doSave);
end

function fig3FinalPerSeed(S, M, figDir, doSave)
% Final policies: one marker per seed, short bar at the mean.
cols = {'err_final_cm', 'Final position error [cm]', '(a) Accuracy'; ...
        'sat_pct', 'Commands at $\pm 1$ [\%]', '(b) Saturation'; ...
        'w_rms', 'Angular rate RMSE [rad/s]', '(c) Angular rate'};
vars = ["initial", "tanh", "tanh_std", "sac"];
ticks = {'Initial', '$\tanh$', '$\tanh+\sigma_{\min}$', 'SAC'};
f = newFigure(S, 16, 5.6);
tl = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for c = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    for i = 1:4
        y = M.(cols{c, 1})(M.variant == vars(i));
        x = i + linspace(-0.08, 0.08, numel(y))';
        if numel(y) == 1, x = i; end
        scatter(ax, x, y, 22, 'MarkerFaceColor', S.variantColor(vars(i), S), 'MarkerEdgeColor', 'w');
        plot(ax, [i - 0.22, i + 0.22], mean(y) * [1 1], '-', 'Color', S.variantColor(vars(i), S), 'LineWidth', 1.6);
    end
    set(ax, 'XTick', 1:4, 'XTickLabel', ticks); xlim(ax, [0.4 4.6]);
    top = max(ylim(ax)); ylim(ax, [-0.04 * top, top]);
    ylabel(ax, cols{c, 2}); title(ax, cols{c, 3}, 'FontWeight', 'normal');
    formatAxis(ax, S); ax.XGrid = 'off';
end
saveFigure(f, figDir, 'fig03_final_per_seed', doSave);
end

function fig4ExampleEpisode(S, X, figDir, doSave)
% One episode with the same initial state and reference: the four rotor commands of each policy and the error.
pols = ["initial", "tanh", "tanh_std", "sac"];
labs = {'PPO, initial', 'PPO, $\tanh$', 'PPO, $\tanh$ + $\sigma_{\min}$', 'SAC'};
f = newFigure(S, 16, 13);
tl = tiledlayout(f, numel(pols) + 1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:numel(pols)
    ax = nexttile(tl); hold(ax, 'on');
    D = X(X.policy == pols(k), :);
    c = S.variantColor(pols(k), S) * 0.85 + 0.15;          % lighter, as the thin traces of the Python figure
    for r = 0:3
        plot(ax, D.t, D.(sprintf('a%d', r)), '-', 'Color', c, 'LineWidth', 0.7);
    end
    ylim(ax, [-1.15 1.15]); yticks(ax, [-1 0 1]); xlim(ax, [0 5]); ylabel(ax, '$a_i$');
    text(ax, 0.995, 0.93, labs{k}, 'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'w', 'Margin', 1);
    formatAxis(ax, S); ax.XTickLabel = [];
end
ax = nexttile(tl); hold(ax, 'on');
h = gobjects(1, numel(pols));
for k = 1:numel(pols)
    D = X(X.policy == pols(k), :);
    h(k) = plot(ax, D.t, D.err, '-', 'Color', S.variantColor(pols(k), S), 'LineWidth', 1.2);
end
set(ax, 'YScale', 'log'); ylim(ax, [0.01 1]); xlim(ax, [0 5]);
xlabel(ax, 'Time [s]'); ylabel(ax, '$\|\mathbf{p}-\mathbf{p}_r\|$ [m]');
formatAxis(ax, S);
legend(ax, h, labs, 'Location', 'northeast', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex');
saveFigure(f, figDir, 'fig04_example_episode', doSave);
end

function fig5Ablation(S, A, figDir, doSave)
% Reward and observation variants: final error against the command variation of every run.
% Colour: velocity cost; marker: attitude term (diamond: angular-rate cost); open marker: seed other than 3.
f = newFigure(S, 12, 7.4);
ax = axes(f); hold(ax, 'on');
ptX = 0.0078; ptY = 0.083;                       % one typographic point in decades (x) and in cm of error (y)
for i = 1:height(A)
    run = char(A.run(i));
    tag = strtok(run, '_');
    withVel = A.w_vel(i) > 0;
    solid = A.seed(i) == 3;
    c = S.orange; if withVel, c = S.blue; end
    shape = 'o';
    if A.ori_cost(i) == "none", shape = '^'; elseif A.ori_cost(i) == "trace", shape = 's'; end
    if A.ang_cost(i) ~= "none", shape = 'd'; end
    face = 'w'; if solid, face = c; end
    plot(ax, A.d_action(i), A.err_final_cm(i), shape, 'Color', c, 'MarkerFaceColor', face, 'MarkerSize', 5.5, ...
        'LineWidth', 1.1);
    [dx, dy, ha] = labelOffset(run);
    label = tag; if ~solid, label = sprintf('%s (s%d)', tag, A.seed(i)); end
    text(ax, A.d_action(i) * 10^(dx * ptX), A.err_final_cm(i) + dy * ptY, label, 'FontSize', 7.5, ...
        'HorizontalAlignment', ha, 'VerticalAlignment', 'middle');
end
set(ax, 'XScale', 'log'); xlim(ax, [8e-4 0.4]); ylim(ax, [0 13]);
xlabel(ax, 'Command variation $\|\Delta\mathbf{a}\|$ per step (last 2\,s)'); ylabel(ax, 'Final position error [cm]');
formatAxis(ax, S);
g = [0.35 0.35 0.35];
hl = gobjects(1, 7);
hl(1) = plot(ax, nan, nan, 'o', 'Color', S.blue, 'MarkerFaceColor', S.blue);
hl(2) = plot(ax, nan, nan, 'o', 'Color', S.orange, 'MarkerFaceColor', S.orange);
hl(3) = plot(ax, nan, nan, 'o', 'Color', g, 'MarkerFaceColor', g);
hl(4) = plot(ax, nan, nan, '^', 'Color', g, 'MarkerFaceColor', g);
hl(5) = plot(ax, nan, nan, 's', 'Color', g, 'MarkerFaceColor', g);
hl(6) = plot(ax, nan, nan, 'd', 'Color', g, 'MarkerFaceColor', g);
hl(7) = plot(ax, nan, nan, 'o', 'Color', g, 'MarkerFaceColor', 'w');
legend(ax, hl, {'With velocity cost', 'Without velocity cost', 'Attitude: $q_z$', 'Attitude: none', ...
    'Attitude: geometric', 'Angular-rate cost', 'Other seed'}, 'Location', 'northeast', 'NumColumns', 2, ...
    'Box', 'off', 'Interpreter', 'latex', 'FontSize', 7);
saveFigure(f, figDir, 'fig05_ablation', doSave);
end

function [dx, dy, ha] = labelOffset(run)
% Label offsets (points) of the runs of the ablation figure; default to the upper right of the marker.
switch run
    case 'B0',     dx = 6;  dy = 0;   ha = 'left';
    case 'B0_s01', dx = 0;  dy = 7;   ha = 'center';
    case 'B0_s02', dx = -6; dy = 0;   ha = 'right';
    case 'R6_s03', dx = 9;  dy = -9;  ha = 'left';
    case 'R1',     dx = 6;  dy = -11; ha = 'left';
    case 'R5',     dx = 6;  dy = -4;  ha = 'left';
    case {'O2_s01', 'O2_s02'}, dx = 6; dy = -4; ha = 'left';
    case 'R3',     dx = 6;  dy = -3;  ha = 'left';
    case 'R7_s03', dx = 9;  dy = 1;   ha = 'left';
    case 'R8_s03', dx = -6; dy = -9;  ha = 'right';
    otherwise,     dx = 6;  dy = 3;   ha = 'left';
end
end

function fig6Lemniscate(S, L, figDir, doSave)
% Figure eight: position RMSE and mean error along the path against the peak acceleration (and speed) of the reference.
% The run with the relaxed speed bound replaces the one with the training bound; a level where most vehicles were lost
% is not drawn.
key = L.policy + "|" + string(L.peak_acc);
[~, ~, g] = unique(key);
keep = false(height(L), 1);
for k = 1:max(g)
    idx = find(g == k);
    [~, j] = max(L.speed_bound(idx));
    keep(idx(j)) = true;
end
Lb = L(keep & L.survival >= 0.5, :);
levels = unique(Lb.peak_acc)';
speed = arrayfun(@(a) mean(Lb.ref_speed_max(Lb.peak_acc == a), 'omitnan'), levels);
isFM = ~cellfun(@isempty, regexp(cellstr(Lb.policy), '^FM_s\d\d$', 'once'));

pol = {'B0', 'SAC', 'PRETRAIN', 'A1', 'A2', 'O1'};
col = {S.sac, S.grey, S.grey, S.blue, S.green, S.orange};
sty = {'-', '--', ':', '-', '-', '-'};
mrk = {'o', 'x', '+', 'o', 's', '^'};
lab = {'PPO baseline', 'SAC', 'RAPTOR pre-training', 'PPO + $\mathbf{a}_r$', ...
       'PPO + $\mathbf{a}_r$ + $\lVert\mathbf{v}-\mathbf{v}_r\rVert$', 'PPO + $\mathbf{a}_r$, geometric'};
panels = {'rmse_cm', 'Position RMSE [cm]', '(a) Tracking error'; ...
          'along_mean_cm', 'Mean error along the path [cm]', '(b) Lag'};

f = newFigure(S, 16, 6.6);
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
hAll = gobjects(1, numel(pol) + 1);
for p = 1:2
    c = panels{p, 1};
    ax = nexttile(tl, p); hold(ax, 'on');
    % range over the three baseline seeds
    lo = nan(size(levels)); hi = nan(size(levels));
    for i = 1:numel(levels)
        v = nan(1, 3);
        for k = 1:3
            sp = {'B0', 'B0_s1', 'B0_s2'};
            r = Lb.(c)(Lb.policy == sp{k} & Lb.peak_acc == levels(i));
            if ~isempty(r), v(k) = r(1); end
        end
        if all(~isnan(v)), lo(i) = min(v); hi(i) = max(v); end
    end
    ok = ~isnan(lo);
    fill(ax, [levels(ok), fliplr(levels(ok))], [lo(ok), fliplr(hi(ok))], S.sac, 'FaceAlpha', 0.18, ...
        'EdgeColor', 'none', 'HandleVisibility', 'off');
    % thrust and moments: median and interquartile range over the seeds that were not lost (at least six)
    fa = []; q = [];
    for i = 1:numel(levels)
        v = Lb.(c)(isFM & Lb.peak_acc == levels(i));
        v = v(~isnan(v));
        if numel(v) >= 6, fa(end + 1) = levels(i); q(end + 1, :) = quantiles(v(:)', [0.25 0.5 0.75]); end %#ok<AGROW>
    end
    if ~isempty(fa)
        fill(ax, [fa, fliplr(fa)], [q(:, 1)', fliplr(q(:, 3)')], S.purple, 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
        hAll(1) = plot(ax, fa, q(:, 2)', '-d', 'Color', S.purple, 'LineWidth', 1.2, 'MarkerSize', 3.5, ...
            'MarkerFaceColor', S.purple);
    end
    for k = 1:numel(pol)
        m = Lb.policy == pol{k};
        [x, order] = sort(Lb.peak_acc(m));
        y = Lb.(c)(m); y = y(order);
        hAll(k + 1) = plot(ax, x, y, [sty{k} mrk{k}], 'Color', col{k}, 'LineWidth', 1.2, 'MarkerSize', 3.8);
    end
    xlim(ax, [0 max(levels) * 1.03]);
    xlabel(ax, 'Peak reference acceleration [m/s$^2$]'); ylabel(ax, panels{p, 2});
    formatAxis(ax, S);
    yl = ylim(ax);
    patch(ax, [0 1.4 1.4 0], [yl(1) yl(1) yl(2) yl(2)], [0.5 0.5 0.5], 'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
    ylim(ax, yl);
    try, uistack(findobj(ax, 'Type', 'patch'), 'bottom'); catch, end
    if p == 2, yline(ax, 0, '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.6, 'HandleVisibility', 'off'); end
    ax2 = addSpeedAxis(tl, ax, levels, speed);
    if isempty(ax2), title(ax, panels{p, 3}, 'FontWeight', 'normal'); else, title(ax2, panels{p, 3}, 'FontWeight', 'normal'); end
end
names = [{'Thrust and moments'}, lab];
lgd = legend(ax, hAll(isgraphics(hAll)), names(isgraphics(hAll)), 'Orientation', 'horizontal', 'Box', 'off', 'Interpreter', 'latex', ...
    'NumColumns', 3, 'FontSize', 7.5);
lgd.Layout.Tile = 'north';
saveFigure(f, figDir, 'fig06_lemniscate', doSave);
end

function ax2 = addSpeedAxis(tl, ax, acc, speed)
% Top axis with the peak reference speed [m/s] at some of the accelerations of the levels; empty if it cannot be made.
ax2 = [];
try
    a2 = axes(tl);
    a2.Layout.Tile = ax.Layout.Tile;
    a2.XAxisLocation = 'top'; a2.Color = 'none'; a2.Box = 'off';
    a2.XLim = ax.XLim; a2.YLim = ax.YLim;
    a2.YTick = []; a2.YColor = 'none';
    sel = ismember(acc, [0.5 2 4 6 8 10]);
    a2.XTick = acc(sel); a2.XTickLabel = compose('%.1f', speed(sel));
    set(a2, 'TickLabelInterpreter', 'latex', 'FontName', 'Computer Modern', 'FontSize', 7, 'TickDir', 'in', ...
        'LineWidth', 0.6);
    a2.XGrid = 'off'; a2.YGrid = 'off';
    xlabel(a2, 'Peak reference speed [m/s]', 'FontSize', 8);
    ax2 = a2;
catch err
    warning('Top speed axis not drawn (%s).', err.message);
end
end

function fig7LemniscatePaths(S, P, figDir, doSave)
% Horizontal path of one vehicle (relative to the centre of the figure eight) and the reference.
levels = unique(P.peak_acc)';
pol = {'FM_s03', 'B0', 'PRETRAIN', 'A1'};
col = {S.purple, S.sac, S.grey, S.blue};
lab = {'Thrust and moments', 'PPO baseline', 'RAPTOR pre-training', 'PPO + $\mathbf{a}_r$'};
f = newFigure(S, 16, 6.4);
tl = tiledlayout(f, 1, numel(levels), 'TileSpacing', 'compact', 'Padding', 'compact');
h = gobjects(1, 5);
for i = 1:numel(levels)
    ax = nexttile(tl, i); hold(ax, 'on');
    m = P.peak_acc == levels(i);
    ref = P(m & P.policy == "B0", :);
    % the reference is a wide light line under the paths with a thin white dashed line on top, so it stays visible
    plot(ax, ref.ref_x, ref.ref_y, '-', 'Color', [0.74 0.74 0.74], 'LineWidth', 5, 'HandleVisibility', 'off');
    for k = 1:numel(pol)
        d = P(m & P.policy == pol{k}, :);
        if isempty(d), continue; end
        h(k + 1) = plot(ax, d.x, d.y, '-', 'Color', col{k}, 'LineWidth', 1.0);
    end
    h(1) = plot(ax, ref.ref_x, ref.ref_y, '--', 'Color', [0.74 0.74 0.74], 'LineWidth', 5);
    plot(ax, ref.ref_x, ref.ref_y, '--', 'Color', 'w', 'LineWidth', 1, 'HandleVisibility', 'off');
    axis(ax, 'equal');
    xlabel(ax, '$x - x_0$ [m]'); ylabel(ax, '$y - y_0$ [m]');
    title(ax, sprintf('$a_{\\mathrm{peak}} = %g$ m/s$^2$', levels(i)), 'FontWeight', 'normal');
    formatAxis(ax, S);
end
valid = isgraphics(h);
names = [{'Reference'}, lab];
lgd = legend(ax, h(valid), names(valid), 'Orientation', 'horizontal', 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 7.5);
lgd.Layout.Tile = 'north';
saveFigure(f, figDir, 'fig07_lemniscate_paths', doSave);
end

%% =============================================================== helpers
function restoreDefaults(names, values)
for k = 1:numel(names), set(groot, names{k}, values{k}); end
end

function q = quantiles(v, p)
% Quantiles of the vector v (linear interpolation; no Statistics Toolbox needed).
v = sort(v(:)');
n = numel(v);
pos = 1 + p * (n - 1);
lo = floor(pos); hi = ceil(pos); w = pos - lo;
q = v(lo) .* (1 - w) + v(hi) .* w;
end

function S = style()
S.font   = 9;
S.blue   = [0.165 0.471 0.839];     % #2a78d6  PPO, initial
S.orange = [0.922 0.408 0.204];     % #eb6834  PPO, tanh
S.green  = [0.106 0.686 0.478];     % #1baf7a  PPO, tanh + sigma_min
S.sac    = [0.251 0.251 0.251];     % #404040  SAC / baseline
S.grey   = [0.541 0.541 0.541];     % #8a8a8a
S.light  = [0.784 0.784 0.784];     % #c8c8c8
S.purple = [0.557 0.267 0.678];     % #8e44ad  thrust and moments
S.variant = [S.blue; S.orange; S.green];
S.variantColor = @(name, st) variantColor(name, st);
end

function c = variantColor(name, S)
switch char(name)
    case 'initial',  c = S.blue;
    case 'tanh',     c = S.orange;
    case 'tanh_std', c = S.green;
    otherwise,       c = S.sac;
end
end

function f = newFigure(S, widthCm, heightCm)
f = figure('Units', 'centimeters', 'Color', 'w', 'WindowStyle', 'normal');
f.Position = [2 2 widthCm heightCm];
set(f, 'DefaultAxesFontSize', S.font);
end

function [f, tl] = twoPanels(S)
f = newFigure(S, 16, 6.0);
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
end

function formatAxis(ax, S)
set(ax, 'FontName', 'Computer Modern', 'FontSize', S.font, 'TickLabelInterpreter', 'latex', 'Box', 'on', ...
    'TickDir', 'in', 'LineWidth', 0.6);
grid(ax, 'on'); ax.GridAlpha = 0.15;
end

function [x, Y] = bySeed(Tab, xField, yField)
% Common x values and one column per seed (seeds may log different numbers of points).
x = unique(Tab.(xField));
x = x(~isnan(x));
seeds = unique(Tab.seed);
Y = nan(numel(x), numel(seeds));
for s = 1:numel(seeds)
    r = Tab(Tab.seed == seeds(s), :);
    [tf, loc] = ismember(r.(xField), x);
    Y(loc(tf), s) = r.(yField)(tf);
end
end

function saveFigure(f, figDir, name, doSave)
% PNG with exportgraphics; PDF with print -painters (same path as the manual export of the figure window):
% the LaTeX text becomes vector outlines instead of selectable text set in a substitute font.
if ~doSave, return; end
drawnow;
if ~isvalid(f), warning('Figure %s was closed before it could be saved.', name); return; end
exportgraphics(f, fullfile(figDir, [name '.png']), 'Resolution', 300);
set(f, 'PaperUnits', 'centimeters');
pos = get(f, 'Position');
set(f, 'PaperSize', pos(3:4), 'PaperPosition', [0 0 pos(3:4)]);
print(f, fullfile(figDir, [name '.pdf']), '-dpdf', '-painters');
end
