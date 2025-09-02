// 31 - 08 - 2025 
// ALNS cho phân vùng shipper: k-means++ kh?i t?o + ALNS (destroy/repair) + ki?m tra liên thông
// B?n này s?a l?i, b? sung 2 destroy m?i, ~5 repair, c?p nh?t w_destroy/w_repair và vòng l?p alns.

#include <bits/stdc++.h>
using namespace std;

#define read() ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define ii pair<int,int>
#define X first
#define Y second

// ===== Tham s? / h?ng =====
const int MAXN = 1000 + 5;     // n t?i da
const int MAXP = 50 + 5;       // p t?i da (theo file g?c dùng N=50)
const double gamma_ = 0.10;    // tr?ng s? gi?a kho?ng cách và penalty cân b?ng

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

long long Rand(long long l,long long r){
    unsigned long long x = ((unsigned long long)rng() << 32) ^ rng();
    return (long long)(x % (unsigned long long)(r - l + 1)) + l;
}

double Rand01(){
    return (double)rng() / (double)rng.max();
}

// ===== Bi?n toàn c?c =====
int n, m, p;                          // s? node, c?nh, s? vùng
int p_centroid[MAXP];                  // p_centroid[k] = node id c?a centroid vùng k (1..p)
int territory[MAXN];                   // territory[i] = ch? s? vùng c?a node i (1..p)

double x_coord[MAXN], y_coord[MAXN];  // to? d? node
// w[][0] = s? don hàng; w[][1] = s? khách hàng
double w[MAXN][2];                     
// u[0], u[1] = trung bình m?i vùng; t[0], t[1] = tolerance
double u[2] = {0.0, 0.0}, tval[2] = {0.0, 0.0};

// d[i][j] = kho?ng cách Euclid gi?a node i và j
static double dmat[MAXN][MAXN];

// danh sách k?, và danh sách node c?a t?ng vùng
vector<int> adjlst[MAXN], node_in[MAXP];

// T?ng theo vùng (d? incremental)
double wsum[MAXP][2];

// ===== Ti?n ích =====

double dist2d(int i,int j){
    double dx = x_coord[i] - x_coord[j];
    double dy = y_coord[i] - y_coord[j];
    return sqrt(dx * dx + dy * dy);
}

void rebuild_node_lists(){
    for(int k = 1; k <= p; k++) node_in[k].clear();
    for(int i = 1; i <= n; i++) if(territory[i] >= 1 && territory[i] <= p) node_in[territory[i]].push_back(i);
}

bool can_attach_to_territory(int k,int j){
    // có ít nh?t 1 hàng xóm dã thu?c vùng k, ho?c vùng k hi?n r?ng
    for(int v : adjlst[j]) if(territory[v] == k) return true;
    return node_in[k].empty();
}

bool remains_connected_after_remove(int k,int rem){
    if((int)node_in[k].size() <= 1) return true;
    vector<int> subset;
    subset.reserve(node_in[k].size());
    for(int v : node_in[k]) if(v != rem) subset.push_back(v);
    if(subset.empty()) return true;
    queue<int> q; q.push(subset[0]);
    unordered_set<int> vis; vis.insert(subset[0]);
    unordered_set<int> S(subset.begin(), subset.end());
    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int v : adjlst[u]) if(S.count(v) && !vis.count(v)){
            vis.insert(v); q.push(v);
        }
    }
    return (vis.size() == S.size());
}

bool connected(int k){
    if(node_in[k].empty()) return true;
    queue<int> q; q.push(node_in[k][0]);
    unordered_set<int> vis; vis.insert(node_in[k][0]);
    unordered_set<int> S(node_in[k].begin(), node_in[k].end());
    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int v : adjlst[u]) if(S.count(v) && !vis.count(v)){
            vis.insert(v); q.push(v);
        }
    }
    return (vis.size() == S.size());
}

// Dùng khi dang thao tác v?i x = (value, vector centroid_of_i)
void rebuild_node_list(pair<double,vector<int>> &x){
    map<int,int> mp; // map node-id centroid -> ch? s? vùng m?i (1..p)
    int cnt = 0;
    for(int k = 1; k <= p; k++) node_in[k].clear();
    for(int i = 0; i < (int)x.Y.size(); i++){
        if(x.Y[i] == -1) continue; // chua gán
        if(!mp.count(x.Y[i])){
            cnt++;
            mp[x.Y[i]] = cnt;
            p_centroid[cnt] = x.Y[i];
        }
        node_in[ mp[x.Y[i]] ].push_back(i + 1);
        territory[i + 1] = mp[x.Y[i]];
    }
}

// ===== Ð?c input & kh?i t?o k-means++ =====

void input(){
    cin >> n;
    for(int i = 1, id, tmp; i <= n; i++){
        cin >> id; id++;
        cin >> x_coord[id] >> y_coord[id] >> w[id][0] >> w[id][1] >> tmp;
        u[0] += w[id][0];
        u[1] += w[id][1];
    }
    cin >> m;
    for(int i = 1, u_, v_; i <= m; i++){
        cin >> u_ >> v_;
        u_++; v_++;
        adjlst[u_].push_back(v_);
        adjlst[v_].push_back(u_);
    }
    int tmp;
    cin >> p >> tmp >> tval[0] >> tval[1];

    // trung bình m?i vùng
    u[0] = u[0] / (double)p;
    u[1] = u[1] / (double)p;

    // fallback n?u input không h?p l?
    if(!(tval[0] > 0 && tval[0] < 1)) tval[0] = 0.05;
    if(!(tval[1] > 0 && tval[1] < 1)) tval[1] = 0.05;

    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= n; j++)
            dmat[i][j] = dist2d(i, j);
}

vector<int> location(){
    vector<bool> chosen (n + 5, 0);
    int first = Rand(1, n);
    p_centroid[1] = first; chosen[first] = 1;
    // k-means++: ch?n p centroid
    for(int c = 2; c <= p; c++){
        int best = 1; double bestd = 1e18;
        for(int i = 1; i <= n; i++) if(!chosen[i]){
            double dd = 0;
            for(int j = 1; j < c; j++) dd = max(dd, dmat[p_centroid[j]][i]);
            if(dd < bestd){ bestd = dd; best = i; }
        }
        p_centroid[c] = best; chosen[best] = 1;
    }
    // gán ban d?u theo centroid g?n nh?t
    for(int i = 1; i <= n; i++){
        int res = 1; double md = 1e18;
        for(int j = 1; j <= p; j++) if(dmat[i][p_centroid[j]] < md){ md = dmat[i][p_centroid[j]]; res = j; }
        territory[i] = res;
    }
    // update centroid theo medoid don gi?n
    int cnt = 0;
    while(true){
        for(int k = 1; k <= p; k++) node_in[k].clear();
        for(int i = 1; i <= n; i++) node_in[territory[i]].push_back(i);
        int changed = 0;
        for(int k = 1; k <= p; k++) if(!node_in[k].empty()){
            double best = 1e18; int res = node_in[k][0];
            for(int a : node_in[k]){
                double sumd = 0; for(int b : node_in[k]) sumd += dmat[a][b];
                if(sumd < best){ best = sumd; res = a; }
            }
            if(p_centroid[k] != res){ p_centroid[k] = res; changed++; }
        }
        for(int i = 1; i <= n; i++){
            int res = territory[i]; double md = dmat[i][p_centroid[territory[i]]];
            for(int j = 1; j <= p; j++) if(dmat[i][p_centroid[j]] < md){ md = dmat[i][p_centroid[j]]; res = j; }
            territory[i] = res;
        }
        cnt++;
        if(changed == 0 || cnt > 50) break;
    }
    vector<int> x_;
    x_.reserve(n);
    for(int i = 1; i <= n; i++) x_.push_back(p_centroid[territory[i]]);
    return x_;
}

// ===== Merit function =====

// f = t?ng kho?ng cách (chu?n hoá theo dmax), g = penalty cân b?ng
// merit = gamma * f + (1 - gamma) * g

double meritfunction(double k_weight = gamma_){
    double f = 0.0, dmax = 0.0;
    for(int i = 1; i <= n; i++){
        dmax = max(dmax, dmat[i][ p_centroid[ territory[i] ] ]);
        f += dmat[i][ p_centroid[ territory[i] ] ];
    }
    if(dmax <= 0) dmax = 1.0;
    f /= dmax;

    double g[2] = {0.0, 0.0};
    for(int ic = 0; ic <= 1; ic++){
        for(int k = 1; k <= p; k++) wsum[k][ic] = 0.0;
        for(int i = 1; i <= n; i++) wsum[ territory[i] ][ic] += w[i][ic];
        for(int k = 1; k <= p; k++){
            double over = max(0.0, wsum[k][ic] - (1 + tval[ic]) * u[ic]);
            double under = max(0.0, (1 - tval[ic]) * u[ic] - wsum[k][ic]);
            double pen = max(over, under);
            g[ic] += (u[ic] > 0) ? (pen / u[ic]) : 0.0;
        }
    }
    double gsum = g[0] + g[1];
    return k_weight * f + (1.0 - k_weight) * gsum;
}

pair<double,vector<int>> get_x(){
    vector<int> x_;
    x_.reserve(n);
    for(int i = 1; i <= n; i++) x_.push_back(p_centroid[ territory[i] ]);
    return { meritfunction(), x_ };
}

// ===== Destroy operators =====

// 1) Biên vùng: xoá các node n?m ? ranh gi?i
vector<int> destroy_on_boundary(){
    int number_of_territory = max(1, (int)Rand(1, max(1, p / 2)));
    vector<int> idx; idx.reserve(p);
    for(int i = 1; i <= p; i++) idx.push_back(i);
    shuffle(idx.begin(), idx.end(), rng);

    vector<int> destroy;
    for(int it = 0; it < number_of_territory; it++){
        int pk = idx[it];
        deque<int> q; q.push_back(p_centroid[pk]);
        vector<int> vis(n + 5, 0);
        vis[p_centroid[pk]] = 1;
        vector<int> boundary;
        while(!q.empty()){
            int u = q.front(); q.pop_front();
            for(int v : adjlst[u]){
                if(vis[v]) continue;
                if(territory[v] != territory[u]){
                    vis[u] = -1; // u là biên
                } else {
                    vis[v] = 1; q.push_back(v);
                }
            }
        }
        for(int i = 1; i <= n; i++) if(vis[i] == -1) boundary.push_back(i);
        shuffle(boundary.begin(), boundary.end(), rng);
        int take = (int)Rand(0, (int)boundary.size() / 3);
        for(int i = 0; i < take; i++) destroy.push_back(boundary[i]);
    }
    return destroy;
}

// 2) C?m ng?u nhiên trong 1 vùng
vector<int> destroy_random_node(){
    vector<int> all; all.reserve(n);
    for(int i = 1; i <= n; i++) all.push_back(i);
    shuffle(all.begin(), all.end(), rng);
    int seed = all[0];
    int kt = territory[seed];
    vector<int> vis(n + 5, 0); deque<int> q; q.push_back(seed); vis[seed] = 1;
    vector<int> comp;
    while(!q.empty()){
        int u = q.front(); q.pop_front(); comp.push_back(u);
        for(int v : adjlst[u]) if(!vis[v] && territory[v] == kt){ vis[v] = 1; q.push_back(v);}    
    }
    shuffle(comp.begin(), comp.end(), rng);
    int rem = max(1, (int)comp.size() / (int)Rand(3, 6));
    vector<int> destroy; destroy.reserve(rem);
    for(int i = 0; i < rem && i < (int)comp.size(); i++) destroy.push_back(comp[i]);
    return destroy;
}

// 3) (M?I) Phá vùng l?n: l?y b?t node t? các vùng có size l?n nh?t
vector<int> destroy_large_territory(){
    vector<pair<int,int>> sz; sz.reserve(p);
    for(int k = 1; k <= p; k++) sz.push_back({ (int)node_in[k].size(), k });
    sort(sz.rbegin(), sz.rend());
    int how_many = max(1, (int)Rand(1, max(1, p / 3)));
    vector<int> destroy;
    for(int i = 0; i < how_many && i < (int)sz.size(); i++){
        int k = sz[i].second;
        vector<int> tmp = node_in[k];
        shuffle(tmp.begin(), tmp.end(), rng);
        int rem = max(1, (int)tmp.size() / (int)Rand(3, 5));
        for(int j = 0; j < rem && j < (int)tmp.size(); j++) destroy.push_back(tmp[j]);
    }
    return destroy;
}

// 4) (M?I) Phá các node xa centroid nh?t (corridor/duôi dài)
vector<int> destroy_far_from_centroid(){
    vector<int> destroy; destroy.reserve(max(1, n / 20));
    for(int k = 1; k <= p; k++) if(!node_in[k].empty()){
        vector<pair<double,int>> a; a.reserve(node_in[k].size());
        for(int v : node_in[k]) a.push_back({ dmat[v][ p_centroid[k] ], v });
        sort(a.rbegin(), a.rend());
        int rem = max(0, (int)a.size() / 5); // l?y top 20% xa nh?t
        int take = (int)Rand(0, rem);
        for(int i = 0; i < take; i++) destroy.push_back(a[i].second);
    }
    return destroy;
}

// ===== Repair operators =====

// 1) Greedy: gán node -1 vào vùng có centroid g?n nh?t mà v?n k?/kh? thi
pair<double,vector<int>> greedy_repair(pair<double,vector<int>> x){
    vector<int> un;
    for(int i = 0; i < (int)x.Y.size(); i++) if(x.Y[i] == -1) un.push_back(i + 1);
    shuffle(un.begin(), un.end(), rng);
    for(int v : un){
        int bestk = -1; double bestd = 1e18;
        for(int k = 1; k <= p; k++){
            if(!can_attach_to_territory(k, v)) continue;
            double dd = dmat[v][ p_centroid[k] ];
            if(dd < bestd){ bestd = dd; bestk = k; }
        }
        if(bestk == -1){ // fallback: g?n nh?t
            for(int k = 1; k <= p; k++){
                double dd = dmat[v][ p_centroid[k] ];
                if(dd < bestd){ bestd = dd; bestk = k; }
            }
        }
        x.Y[v - 1] = p_centroid[bestk];
    }
    rebuild_node_list(x);
    double val = meritfunction();
    x.X = val;
    return x;
}

// 2) Random (uu tiên các vùng k?)
pair<double,vector<int>> random_repair(pair<double,vector<int>> x){
    vector<int> un; for(int i = 0; i < (int)x.Y.size(); i++) if(x.Y[i] == -1) un.push_back(i + 1);
    for(int v : un){
        vector<int> cand;
        for(int k = 1; k <= p; k++) if(can_attach_to_territory(k, v)) cand.push_back(k);
        int bestk;
        if(cand.empty()){
            bestk = 1; double bestd = dmat[v][ p_centroid[1] ];
            for(int k = 2; k <= p; k++) if(dmat[v][ p_centroid[k] ] < bestd){ bestd = dmat[v][ p_centroid[k] ]; bestk = k; }
        } else {
            bestk = cand[ Rand(0, (int)cand.size() - 1) ];
        }
        x.Y[v - 1] = p_centroid[bestk];
    }
    rebuild_node_list(x);
    x.X = meritfunction();
    return x;
}

// 3) Best-balance: gi?m l?ch cân b?ng + xét kho?ng cách
pair<double,vector<int>> best_balance_repair(pair<double,vector<int>> x){
    vector<int> un; for(int i = 0; i < (int)x.Y.size(); i++) if(x.Y[i] == -1) un.push_back(i + 1);
    // current weights per territory
    vector<array<double,2>> cur(p + 1, {0.0, 0.0});
    for(int k = 1; k <= p; k++) for(int v : node_in[k]){ cur[k][0] += w[v][0]; cur[k][1] += w[v][1]; }

    for(int v : un){
        int bestk = 1; double bestscore = 1e18;
        for(int k = 1; k <= p; k++){
            double nw0 = cur[k][0] + w[v][0];
            double nw1 = cur[k][1] + w[v][1];
            double pen0 = max({ nw0 - (1 + tval[0]) * u[0], (1 - tval[0]) * u[0] - nw0, 0.0 });
            double pen1 = max({ nw1 - (1 + tval[1]) * u[1], (1 - tval[1]) * u[1] - nw1, 0.0 });
            double dd = dmat[v][ p_centroid[k] ];
            double score = ( (u[0] > 0 ? pen0 / u[0] : 0) + (u[1] > 0 ? pen1 / u[1] : 0) ) + 0.01 * dd;
            if(!can_attach_to_territory(k, v)) score *= 1.2; // ph?t nh? n?u không k?
            if(score < bestscore){ bestscore = score; bestk = k; }
        }
        x.Y[v - 1] = p_centroid[bestk];
        cur[bestk][0] += w[v][0]; cur[bestk][1] += w[v][1];
    }
    rebuild_node_list(x);
    x.X = meritfunction();
    return x;
}

// 4) BFS region growing: d?m b?o liên thông tru?c
pair<double,vector<int>> bfs_repair(pair<double,vector<int>> x){
    vector<int> assigned(n + 1, 0);
    for(int i = 1; i <= n; i++) if(x.Y[i - 1] != -1) assigned[i] = 1;
    queue<int> q;
    for(int k = 1; k <= p; k++){
        int c = p_centroid[k];
        if(!assigned[c]){ x.Y[c - 1] = p_centroid[k]; assigned[c] = 1; }
        q.push(c);
    }
    while(!q.empty()){
        int u = q.front(); q.pop();
        int terr = territory[u]; if(terr < 1 || terr > p) continue;
        for(int v : adjlst[u]) if(!assigned[v]){
            x.Y[v - 1] = p_centroid[terr];
            assigned[v] = 1; q.push(v);
        }
    }
    for(int i = 1; i <= n; i++) if(x.Y[i - 1] == -1){
        int bestk = 1; double bd = dmat[i][ p_centroid[1] ];
        for(int k = 2; k <= p; k++) if(dmat[i][ p_centroid[k] ] < bd){ bd = dmat[i][ p_centroid[k] ]; bestk = k; }
        x.Y[i - 1] = p_centroid[bestk];
    }
    rebuild_node_list(x);
    x.X = meritfunction();
    return x;
}

// 5) Priority-penalty (hàng d?i uu tiên)
pair<double,vector<int>> priority_penalty_repair(pair<double,vector<int>> x){
    vector<int> un; for(int i = 0; i < (int)x.Y.size(); i++) if(x.Y[i] == -1) un.push_back(i + 1);
    vector<array<double,2>> cur(p + 1, {0.0, 0.0});
    for(int k = 1; k <= p; k++) for(int v : node_in[k]){ cur[k][0] += w[v][0]; cur[k][1] += w[v][1]; }

    struct Cand{ double cost; int v; int k; };
    struct Cmp{ bool operator()(const Cand &a, const Cand &b) const { return a.cost > b.cost; } };
    priority_queue<Cand, vector<Cand>, Cmp> pq;

    for(int v : un){
        for(int k = 1; k <= p; k++){
            double nw0 = cur[k][0] + w[v][0];
            double nw1 = cur[k][1] + w[v][1];
            double pen0 = max({ nw0 - (1 + tval[0]) * u[0], (1 - tval[0]) * u[0] - nw0, 0.0 });
            double pen1 = max({ nw1 - (1 + tval[1]) * u[1], (1 - tval[1]) * u[1] - nw1, 0.0 });
            double dd = dmat[v][ p_centroid[k] ];
            double cost = ( (u[0] > 0 ? pen0 / u[0] : 0) + (u[1] > 0 ? pen1 / u[1] : 0) ) + 0.01 * dd;
            if(!can_attach_to_territory(k, v)) cost *= 1.3;
            pq.push({ cost, v, k });
        }
    }

    vector<int> done(n + 1, 0);
    while(!pq.empty()){
        auto c = pq.top(); pq.pop();
        if(done[c.v]) continue;
        x.Y[c.v - 1] = p_centroid[c.k];
        done[c.v] = 1; cur[c.k][0] += w[c.v][0]; cur[c.k][1] += w[c.v][1];
    }
    rebuild_node_list(x);
    x.X = meritfunction();
    return x;
}

// ===== ALNS framework =====

// Tr?ng s? ch?n operator
vector<double> w_destroy = { 1.0, 1.0, 1.0, 1.0 }; // boundary, random_cluster, large_territory, far_from_centroid
vector<double> w_repair  = { 1.0, 1.0, 1.0, 1.0, 1.0 }; // greedy, random, best_balance, bfs, priority

int pick_by_weight(vector<double> &w){
    double S = 0.0; for(double x : w) S += x;
    if(S <= 0) return (int)Rand(0, (int)w.size() - 1);
    double r = Rand01() * S; double acc = 0.0;
    for(int i = 0; i < (int)w.size(); i++){
        acc += w[i]; if(r <= acc) return i;
    }
    return (int)w.size() - 1;
}

void update_w(int id_d, int id_r, double reward, double rho = 0.8){
    w_destroy[id_d] = w_destroy[id_d] * rho + (1.0 - rho) * reward;
    w_repair [id_r] = w_repair [id_r] * rho + (1.0 - rho) * reward;
}

pair<double,vector<int>> apply_destroy(int id, pair<double,vector<int>> x){
    vector<int> removed;
    if(id == 0) removed = destroy_on_boundary();
    else if(id == 1) removed = destroy_random_node();
    else if(id == 2) removed = destroy_large_territory();
    else if(id == 3) removed = destroy_far_from_centroid();

    for(int v : removed) x.Y[v - 1] = -1;
    rebuild_node_list(x);
    x.X = meritfunction();
    return x;
}

pair<double,vector<int>> apply_repair(int id, pair<double,vector<int>> x){
    if(id == 0) return greedy_repair(x);
    if(id == 1) return random_repair(x);
    if(id == 2) return best_balance_repair(x);
    if(id == 3) return bfs_repair(x);
    if(id == 4) return priority_penalty_repair(x);
    return greedy_repair(x);
}

bool feasible(pair<double,vector<int>> &x){
    rebuild_node_list(x);
    for(int k = 1; k <= p; k++) if(!connected(k)) return false;
    return true;
}

pair<double,vector<int>> x_best, x_cur;

pair<double,vector<int>> alns(int iters = 2000){
    x_best.X = 1e18; x_cur = get_x();
    double best_val = x_cur.X;
    for(int it = 0; it < iters; it++){
        int id_d = pick_by_weight(w_destroy);
        int id_r = pick_by_weight(w_repair);

        auto x1 = apply_destroy(id_d, x_cur);
        auto x2 = apply_repair(id_r, x1);

        bool feas = feasible(x2);
        double reward = 0.01; // m?c d?nh r?t nh?
        if(feas){
            if(x2.X < x_cur.X){
                // c?i thi?n current
                reward = 1.0 + max(0.0, (x_cur.X - x2.X) / max(1.0, x_cur.X));
                x_cur = x2;
            } else {
                // ch?p nh?n t?i hon v?i xác su?t nh? (d? khám phá)
                double prob = 0.05;
                if(Rand01() < prob) x_cur = x2;
                reward = 0.10;
            }
            if(x2.X < x_best.X){
                // c?i thi?n k? l?c
                x_best = x2;
                reward = 5.0 + max(0.0, (x_best.X - x2.X));
            }
        }
        update_w(id_d, id_r, reward, 0.85);
    }
    return x_best;
}

bool print_output(){
    for(int k = 1; k <= p; k++) if(!connected(k)){
        cout << "No solution found for the connected graph!\nHowever, this is the best solution under current constraints:\n";
        break;
    }
    cout << fixed << setprecision(6);
    cout << "Accepted Range of orders for each territory : "
         << (1 - tval[0]) * u[0] << " " << (1 + tval[0]) * u[0] << "\n";
    cout << "Accepted Range of customers for each territory : "
         << (1 - tval[1]) * u[1] << " " << (1 + tval[1]) * u[1] << "\n";

    double tot_dist = 0.0; int cnt_viol = 0;
    for(int k = 1; k <= p; k++){
        cout << "Centroid " << k << " at node " << p_centroid[k] << ": ";
        int w0 = 0, w1 = 0; double dsum = 0.0, radius = 0.0;
        for(int v : node_in[k]){
            cout << v << " "; w0 += w[v][0]; w1 += w[v][1];
            radius = max(radius, dmat[p_centroid[k]][v]);
            dsum += dmat[p_centroid[k]][v];
        }
        bool viol = (w0 < (1 - tval[0]) * u[0] || w0 > (1 + tval[0]) * u[0] ||
                     w1 < (1 - tval[1]) * u[1] || w1 > (1 + tval[1]) * u[1]);
        if(viol) cnt_viol++;
        tot_dist += dsum;
        cout << "\n1 - Orders \t= " << w0
             << "\n2 - Customers \t= " << w1
             << "\n3 - Radius \t= " << radius
             << "\n4 - Distant \t= " << dsum << '\n';
    }
    cout << tot_dist << " - " << cnt_viol << " - " << meritfunction() << '\n';
    return true;
}

int main(){
    read();
    //freopen("input\\2DU60-05-1.dat","r",stdin);
    input();
    location();
    rebuild_node_lists();
    auto best = alns(100000);
    (void)best; // không dùng cung không warning
    print_output();
    return 0;
}

