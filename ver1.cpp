// 23 - 12 - 23 

#include<bits/stdc++.h>

using namespace std;

#define read() ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define day() time_t now = time(0);char* x = ctime(&now);cerr<<"Right Now Is : "<<x<<"\n"

#define ii pair<int,int>
#define X first
#define Y second 

const long long MAX = (int)1000 + 5;
const long long INF = (int)1e9;
const long long MOD = (int)1e9 + 7;
const long long N = 50;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

long long Rand(long long l,long long r){
    long long x = rng() % (long long)1e18;
    return x % (r - l + 1) + l;
}
int n,m,p,p_centroid[N],territory[MAX];
// p_centroid -> centroid p_th placed on node i 
// territory[i] -> i is assigned to th centroid
double x_coord[MAX],y_coord[MAX],w[MAX][2],u[] = {0,0},t[] = {0,0},d[MAX][MAX],wsum[N][2];
vector<int> adj[MAX],node[N];
double dist(int i,int j){
	return sqrt((x_coord[i] - x_coord[j]) * (x_coord[i] - x_coord[j]) + (y_coord[i] - y_coord[j]) * (y_coord[i] - y_coord[j]));
}
void input(){
	cin >> n;
	for(int i = 1,id,tmp;i <= n;i++){
		cin >> id;
		id++;
		cin >> x_coord[id] >> y_coord[id] >> w[id][0] >> w[id][1] >> tmp; 
		u[0] += w[id][0];
		u[1] += w[id][1];
	}
	cin >> m;
	for(int i = 1,u,v;i <= m;i++){
		cin >> u >> v;
		u++,v++;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	cin >> p >> p >> t[0] >> t[1];
	u[0] = u[0] / (double)(p);
	u[1] = u[1] / (double)(p);
	for(int i = 1;i <= n;i++){
		for(int j = 1;j <= n;j++){
			d[i][j] = dist(i,j);
		}
	}
}
double meritfunction(double gamma = 1000,double beta = 500){
	double f = 0;
	for(int i = 1;i <= n;i++)f += d[i][p_centroid[territory[i]]];
	double g[2] = {0,0};
	for(int ic = 0;ic <= 1;ic++){
		for(int i = 1;i <= p;i++){
			wsum[i][ic] = 0;
		}
		for(int i = 1;i <= n;i++){
			wsum[territory[i]][ic] += w[i][ic];
		}
		for(int i = 1;i <= p;i++){
			g[ic] += max({wsum[i][ic] - (1 + t[ic]) * u[ic],(1 - t[ic]) * u[ic] - wsum[i][ic],0.0});
		}
	}
	double g_ = g[0] + g[1];
	return f + gamma * g_;
}
void location(){
	
	vector<bool> choosen (n + 5,0);
	int first = Rand(1, n);
	first = min(23,n);
	p_centroid[1] = first;
	choosen[first] = 1;
	// k-means++
	for(int c = 2;c <= p;c++){
		int res = 0;
		double mindist = (double)1e18;
	   	for(int i = 1;i <= n;i++){
	   		if(choosen[i])continue;
	   		double total_dist = 0;
	   		for(int j = 1;j < c;j++){
	   			total_dist = max(total_dist,d[p_centroid[j]][i]);
	   		}
	   		if(total_dist < mindist){
	   			res = i;
	   			mindist = total_dist;
	   		}
	   	}
		choosen[res] = 1;
		p_centroid[c] = res;
	}
	for(int i = 1;i <= n;i++){
		int res = -1;
		double mindist = (double)1e18;
		for(int j = 1;j <= p;j++){
			if(d[i][p_centroid[j]] < mindist){
				mindist = d[i][p_centroid[j]];
				res = j;
			}
		}
		territory[i] = res;
	}
	
	for(int i = 1;i <= p;i++){
		cout << p_centroid[i] << " \n"[i == p];
	}
	// Gán ngẫu nhiên p centroid
	
	int time_change = 0;
	int cnt = 0;
	while(true){
		time_change = 0;
		for(int i = 1;i <= p;i++)node[i].clear();
		for(int i = 1;i <= n;i++){
			node[territory[i]].push_back(i);
		}
		for(int i = 1;i <= p;i++){
			double best = (double)1e18;
			int res = 0;
			//cout << node[i].size() << '\n';
			for(int ix = 0;ix < (int)(node[i].size());ix++){
				double distant = 0;
				for(int j = 0;j < (int)node[i].size();j++){
					distant += d[node[i][ix]][node[i][j]];
				}
				if(distant < best){
					best = distant;
					res = ix;
				}
			}
			p_centroid[i] = node[i][res];
		}
		for(int i = 1;i <= n;i++){
			int res = territory[i];
			double mindist = d[i][p_centroid[territory[i]]];
			for(int j = 1;j <= p;j++){
				if(d[i][p_centroid[j]] < mindist){
					time_change++;
					mindist = d[i][p_centroid[j]];
					res = j;
				}
			}
			
			territory[i] = res;
		}
		cnt++;
		if(!time_change || cnt > 100)break;
	}
	for(int i = 1;i <= p;i++){
		cout << p_centroid[i] << " \n"[i == p];
	}
}

/*
	Phase 2 : Allocation
	Giải bài toán LP bằng cách tolerant ban đầu = 0 với mỗi tiêu chí 
	Xử lí split node 

*/
pair<double, vector<double>> simplex(vector<double> c, vector<vector<double>> A, vector<double> b, vector<char> sense) {
    int m = A.size(), n = c.size();
    int slackCount = 0;
    for(char s : sense) if(s == '<') slackCount++;
    int cols = n + slackCount + 1;
    vector<vector<double>> tab(m+1, vector<double>(cols, 0));
    int slackPos = n;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) tab[i][j] = A[i][j];
        if(sense[i] == '<') tab[i][slackPos++] = 1;
        tab[i][cols-1] = b[i];
    }
    for(int j = 0; j < n; j++) tab[m][j] = -c[j];
    while(true) {
        int col = min_element(tab[m].begin(), tab[m].end()-1) - tab[m].begin();
        if(tab[m][col] >= -1e-9) break;
        double best = 1e18;
        int row = -1;
        for(int i = 0; i < m; i++) {
            if(tab[i][col] > 1e-9) {
                double ratio = tab[i][cols-1] / tab[i][col];
                if(ratio < best) {
                    best = ratio;
                    row = i;
                }
            }
        }
        if(row == -1) throw runtime_error("Unbounded solution.");
        double div = tab[row][col];
        for(int j = 0; j < cols; j++) tab[row][j] /= div;
        for(int i = 0; i <= m; i++) {
            if(i == row) continue;
            double factor = tab[i][col];
            for(int j = 0; j < cols; j++) tab[i][j] -= factor * tab[row][j];
        }
    }
    vector<double> x(n, 0);
    for(int j = 0; j < n; j++) {
        int one_row = -1;
        bool is_basic = true;
        for(int i = 0; i < m; i++) {
            if(fabs(tab[i][j] - 1) < 1e-9) {
                if(one_row == -1) one_row = i;
                else is_basic = false;
            } else if(fabs(tab[i][j]) > 1e-9) {
                is_basic = false;
            }
        }
        if(is_basic && one_row != -1) x[j] = tab[one_row][cols-1];
    }
    return {tab[m][cols-1], x};
}
pair<double, vector<double>> simplex_criterion(int criterion) {
    int vars = p * n;
    vector<double> c(vars, 0);
    for(int i = 0; i < p; i++) {
        for(int j = 0; j < n; j++) {
            c[i*n + j] = d[p_centroid[i+1]][j+1];
        }
    }
    vector<vector<double>> A;
    vector<double> b;
    vector<char> sense;
    for(int j = 0; j < n; j++) {
        vector<double> row(vars, 0);
        for(int i = 0; i < p; i++) row[i*n + j] = 1;
        A.push_back(row);
        b.push_back(1);
        sense.push_back('=');
    }
    for(int i = 0; i < p; i++) {
        vector<double> row(vars, 0);
        for(int j = 0; j < n; j++) {
            row[i*n + j] = w[j+1][criterion];
        }
        A.push_back(row);
        b.push_back(u[criterion]);
        sense.push_back('=');
    }
    return simplex(c, A, b, sense);
}
// 
// void allocation() {
	// vector<int> splitnodes;
	// vector<double> last(n + 5,-1);
	// vector<bool> choosen(n + 5,0);
    // for(int a = 0; a < 2; a++) {
        // auto res = simplex_criterion(a);
        // double optimum = res.first;
        // vector<double> sol = res.second;
        // cout << "=== Criterion " << a << " ===\n";
        // cout << "Optimum value = " << optimum << "\n";
        // cout << "Solution matrix (p x n):\n";
        // for(int i = 0; i < p; i++) {
            // for(int j = 0; j < n; j++) {
                // double val = sol[i*n + j];
                // if(fabs(val) < 1e-9) val = 0;
                // if((abs(val) > 1e-9) && abs(val - 1) > 1e-9){
                	// if(!choosen[j]){
                		// choosen[j] = 1;
                		// split_nodes.push_back(j + 1);
                	// }
                // }
                // if(abs(val - 1) < 1e-9){
                	// if(a == 0)last[j] = i;
                	// else if(last[j] != i && choosen[j] == 0){
                		// choosen[j] = 1;
                		// split_nodes.push_back(j + 1);
                	// }
                // }
                // cout << setw(8) << fixed << setprecision(3) << val << " ";
            // }
            // cout << "\n";
        // }
    // }
    // for(auto v : split_nodes)cout << v << " ";
//     
// }
void rebuild_node_lists(){
	for(int i = 1;i <= p;i++) node[i].clear();
	for(int j = 1;j <= n;j++){
		int k = territory[j];
		if(k >= 1 && k <= p) node[k].push_back(j);
	}
}

bool can_attach_to_territory(int k,int j){
	for(int v : adj[j]) if(territory[v] == k) return 1;
	return node[k].empty();
}

bool remains_connected_after_remove(int k,int rem){
	if((int)node[k].size() <= 1) return 1;
	vector<int> subset;
	for(int v : node[k]) if(v != rem) subset.push_back(v);
	if(subset.empty()) return 1;
	queue<int> q;q.push(subset[0]);
	unordered_set<int> vis;vis.insert(subset[0]);
	unordered_set<int> S(subset.begin(),subset.end());
	while(!q.empty()){
		int u = q.front();q.pop();
		for(int v : adj[u]) if(S.count(v) && !vis.count(v)){
			vis.insert(v);q.push(v);
		}
	}
	return (vis.size() == S.size());
}

void allocation(){
	auto sol0 = simplex_criterion(0);
	auto sol1 = simplex_criterion(1);
	vector<double> x0 = sol0.second,x1 = sol1.second;
	for(int j = 1;j <= n;j++){
		int bestk = 1;double bestv = -1;
		for(int i = 1;i <= p;i++){
			double v = x0[(i - 1) * n + (j - 1)] + x1[(i - 1) * n + (j - 1)];
			if(v > bestv){bestv = v;bestk = i;}
		}
		territory[j] = bestk;
	}
	rebuild_node_lists();
	vector<vector<int>> cand(n + 1);
	auto collect = [&](const vector<double> &xx){
		for(int j = 1;j <= n;j++){
			int cnt = 0;
			for(int i = 1;i <= p;i++) if(xx[(i - 1) * n + (j - 1)] > 1e-9) cnt++;
			if(cnt >= 2){
				for(int i = 1;i <= p;i++) if(xx[(i - 1) * n + (j - 1)] > 1e-9) cand[j].push_back(i);
			}
		}
	};
	collect(x0);collect(x1);
	for(int j = 1;j <= n;j++){
		int a0 = 1,a1 = 1;double b0 = -1,b1 = -1;
		for(int i = 1;i <= p;i++){
			double v0 = x0[(i - 1) * n + (j - 1)];if(v0 > b0){b0 = v0;a0 = i;}
			double v1 = x1[(i - 1) * n + (j - 1)];if(v1 > b1){b1 = v1;a1 = i;}
		}
		if(a0 != a1){cand[j].push_back(a0);cand[j].push_back(a1);}
		sort(cand[j].begin(),cand[j].end());
		cand[j].erase(unique(cand[j].begin(),cand[j].end()),cand[j].end());
	}
	static double wsum[MAX][2];
	for(int k = 1;k <= p;k++){wsum[k][0] = 0;wsum[k][1] = 0;}
	for(int k = 1;k <= p;k++) for(int j : node[k]){
		wsum[k][0] += w[j][0];wsum[k][1] += w[j][1];
	}
	auto within_balance = [&](int k,int j,int dir){
		for(int a = 0;a < 2;a++){
			double val = wsum[k][a] + dir * w[j][a];
			double up = (1 + t[a]) * u[a],lo = (1 - t[a]) * u[a];
			if(val < lo - 1e-9 || val > up + 1e-9) return 0;
		}
		return 1;
	};
	auto infeas_after_add = [&](int k,int j){
		double total = 0;
		for(int a = 0;a < 2;a++){
			double val = wsum[k][a] + w[j][a];
			double up = (1 + t[a]) * u[a],lo = (1 - t[a]) * u[a];
			double viol = max({val - up,lo - val,0.0});
			total += viol;
		}
		return total;
	};
	vector<int> order;
	for(int j = 1;j <= n;j++) if(!cand[j].empty()) order.push_back(j);
	for(int jj : order){
		int cur = territory[jj];
		if(cand[jj].empty()) continue;
		int bestk = -1;double bestCost = 1e18;bool picked = 0;
		for(int k = 1;k <= p;k++){
			if(!binary_search(cand[jj].begin(),cand[jj].end(),k)) continue;
			if(k == cur) continue;
			if(!can_attach_to_territory(k,jj)) continue;
			if(!within_balance(k,jj,+1)) continue;
			if(!within_balance(cur,jj,-1)) continue;
			double cost = d[p_centroid[k]][jj];
			if(cost < bestCost){bestCost = cost;bestk = k;picked = 1;}
		}
		if(!picked){
			double bestInf = 1e18;bestCost = 1e18;bestk = -1;
			for(int k = 1;k <= p;k++){
				if(!binary_search(cand[jj].begin(),cand[jj].end(),k)) continue;
				if(k == cur) continue;
				if(!can_attach_to_territory(k,jj)) continue;
				double infv = infeas_after_add(k,jj);
				double cost = d[p_centroid[k]][jj];
				if(infv < bestInf - 1e-12 || (fabs(infv - bestInf) <= 1e-12 && cost < bestCost)){
					bestInf = infv;bestCost = cost;bestk = k;
				}
			}
		}
		if(bestk != -1 && bestk != cur){
			auto &vc = node[cur];
			vc.erase(remove(vc.begin(),vc.end(),jj),vc.end());
			node[bestk].push_back(jj);
			territory[jj] = bestk;
			wsum[cur][0] -= w[jj][0];wsum[cur][1] -= w[jj][1];
			wsum[bestk][0] += w[jj][0];wsum[bestk][1] += w[jj][1];
		}
	}
	rebuild_node_lists();
	for(int k = 1;k <= p;k++) if(!node[k].empty()){
		int bestNode = node[k][0];double best = 1e18;
		for(int candn : node[k]){
			double s = 0;for(int j : node[k]) s += d[candn][j];
			if(s < best){best = s;bestNode = candn;}
		}
		p_centroid[k] = bestNode;
	}
}

void local_search(double gamma = 1000.0,int limit_moves = 10000){
	rebuild_node_lists();
	double cur = meritfunction(gamma);
	static double wsum[MAX][2];
	for(int k = 1;k <= p;k++){wsum[k][0] = 0;wsum[k][1] = 0;}
	for(int k = 1;k <= p;k++) for(int j : node[k]){
		wsum[k][0] += w[j][0];wsum[k][1] += w[j][1];
	}
	auto g_val = [&](double val,int a){double up = (1 + t[a]) * u[a],lo = (1 - t[a]) * u[a];return max({val - up,lo - val,0.0});};
	int moves = 0;
	while(true){
		double bestDelta = 0;int bestJ = -1,bestFrom = -1,bestTo = -1;
		for(int j = 1;j <= n;j++){
			int fromK = territory[j];
			for(int v : adj[j]){
				int toK = territory[v];
				if(toK == fromK) continue;
				if(!can_attach_to_territory(toK,j)) continue;
				if(!remains_connected_after_remove(fromK,j)) continue;
				double Fdelta = d[j][p_centroid[toK]] - d[j][p_centroid[fromK]];
				double Gbefore = 0,Gafter = 0;
				for(int a = 0;a < 2;a++){
					double vFrom = wsum[fromK][a],vTo = wsum[toK][a];
					Gbefore += g_val(vFrom,a) + g_val(vTo,a);
					Gafter += g_val(vFrom - w[j][a],a) + g_val(vTo + w[j][a],a);
				}
				double delta = Fdelta + gamma * (Gafter - Gbefore);
				if(delta < bestDelta - 1e-12 || (fabs(delta - bestDelta) <= 1e-12 && (bestJ == -1 || j < bestJ))){
					bestDelta = delta;bestJ = j;bestFrom = fromK;bestTo = toK;
				}
			}
		}
		if(bestJ == -1 || bestDelta >= -1e-12) break;
		node[bestFrom].erase(remove(node[bestFrom].begin(),node[bestFrom].end(),bestJ),node[bestFrom].end());
		node[bestTo].push_back(bestJ);
		for(int a = 0;a < 2;a++){wsum[bestFrom][a] -= w[bestJ][a];wsum[bestTo][a] += w[bestJ][a];}
		territory[bestJ] = bestTo;
		cur += bestDelta;
		moves++;if(moves >= limit_moves) break;
	}
}

void print_territories(){
	rebuild_node_lists();
	for(int k = 1;k <= p;k++){
		cout << "Centroid " << k << " at node " << p_centroid[k] << ": ";
		for(int v : node[k]) cout << v << " ";
		cout << "\n";
	}
	cout << meritfunction();
}



signed main(){
	
	read();
	input();
	location();
	allocation();
	local_search();
	print_territories();
	
}