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
const double gamma_ = 0.1;

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
	int tmp;
	cin >> p >> tmp >> t[0] >> t[1];
	u[0] = u[0] / (double)(p);
	u[1] = u[1] / (double)(p);
	t[0] = t[1] = 0.05;
	for(int i = 1;i <= n;i++){
		for(int j = 1;j <= n;j++){
			d[i][j] = dist(i,j);
		}
	}
}
double meritfunction(double k_weight = gamma_){
	double f = 0;
	double dmax = 0;
	for(int i = 1;i <= n;i++){
		dmax = max(dmax,d[i][p_centroid[territory[i]]]);
		f += d[i][p_centroid[territory[i]]];
	}
	f /= dmax;
	double g[2] = {0,0};
	for(int ic = 0;ic <= 1;ic++){
		for(int i = 1;i <= p;i++){
			wsum[i][ic] = 0;
		}
		for(int i = 1;i <= n;i++){
			wsum[territory[i]][ic] += w[i][ic];
		}
		for(int i = 1;i <= p;i++){
			double x = max({wsum[i][ic] - (1 + t[ic]) * u[ic],(1 - t[ic]) * u[ic] - wsum[i][ic],0.0});
			g[ic] += (u[ic] > 0) ? x / u[ic] : 0;
		}
	}
	double g_ = g[0] + g[1];
	//cout << f << " " << gamma_ << '\n';
	return f * k_weight + (1 - k_weight) * g_;
}
void location(int c = 23){
	vector<bool> choosen (n + 5,0);
	int first = Rand(1,n);
	//cout << first << "\n";
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
	
	// for(int i = 1;i <= p;i++){
		// cout << p_centroid[i] << " \n"[i == p];
	// }
	// Gán ng?u nhiên p centroid
	
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
	// for(int i = 1;i <= p;i++){
		// cout << p_centroid[i] << " \n"[i == p];
	// }
	// exit(0);
}

/*
	Phase 2 : Allocation
	Gi?i bài toán LP b?ng cách tolerant ban d?u = 0 v?i m?i tiêu chí 
	X? lí split node 

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
bool connected(int k){
	vector<int> subset;
	//cout << node[k].size() << "\n";
	for(int v : node[k])subset.push_back(v);
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
int repair_connectivity(int x){
	for(auto k : adj[x]){
		if(territory[k] == territory[x]) continue;	
		if(!remains_connected_after_remove(territory[x],x))continue;
		if(connected(territory[k]) || territory[k] > p || territory[k] < 1)continue;
		int last = territory[x];
		territory[x] = territory[k];
		//cout << k << '\n';
		if(connected(territory[k])){
			return k;
		}
		territory[x] = last;
	}
	return -1;
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
			// cand[j] luu nh?ng centroid có kh? nang % 
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
		// cand[j] . 2 tiêu chí ch?n 2 centroid khác nhau
		sort(cand[j].begin(),cand[j].end());
		cand[j].erase(unique(cand[j].begin(),cand[j].end()),cand[j].end());
	}
	static double wsum[MAX][2];
	for(int k = 1;k <= p;k++){wsum[k][0] = 0;wsum[k][1] = 0;}
	for(int k = 1;k <= p;k++) for(int j : node[k]){
		wsum[k][0] += w[j][0];wsum[k][1] += w[j][1];
	}
	auto within_balance = [&](int k,int j,int dir){
		// dir = {-1,1} -> ktra xem có cân b?ng n?u thêm ho?c b?t j t? k hay không 
		for(int a = 0;a < 2;a++){
			double val = wsum[k][a] + dir * w[j][a];
			double up = (1 + t[a]) * u[a],lo = (1 - t[a]) * u[a];
			if(val < lo - 1e-9 || val > up + 1e-9) return 0;
		}
		return 1;
	};
	auto infeas_after_add = [&](int k,int j){
		// tính toán t?ng vi ph?m thay d?i n?u thêm j vào k 
		double total = 0;
		for(int a = 0;a < 2;a++){
			double val = wsum[k][a] + w[j][a];
			double up = (1 + t[a]) * u[a],lo = (1 - t[a]) * u[a];
			double viol = max({val - up,lo - val,0.0});
			total += viol;
		}
		return total;
	};
	
	vector<int> split_nodes;
	for(int j = 1;j <= n;j++) if(!cand[j].empty()) split_nodes.push_back(j);
	vector<int> choosen;
	for(int i = 1;i <= 100;i++){
		if(split_nodes.empty())break;
		bool ok = 1;
		for(int j = 1;j <= p;j++){
			if(!connected(j)){
				ok = 0;
				break;
			}
		}
		if(!ok)break;
		for(int i = 0;i < (int)split_nodes.size();i++){
			int k = repair_connectivity(split_nodes[i]);
			if(k == -1){
				choosen.push_back(split_nodes[i]);
				continue;
			}
			//cout << k << " - " << territory[k] << '\n';
			auto &vc = node[territory[split_nodes[i]]];
			vc.erase(remove(vc.begin(),vc.end(),split_nodes[i]),vc.end());
			node[territory[k]].push_back(split_nodes[i]);
			territory[split_nodes[i]] = territory[k];
		}	
		rebuild_node_lists();
		split_nodes = choosen;
		choosen.clear();
	}
	
	for(int i = 1;i <= 100;i++){
		if(split_nodes.empty())break;
		for(int i = 0;i < (int)split_nodes.size();i++){
			int k = repair_connectivity(split_nodes[i]);
			if(k == -1){
				choosen.push_back(split_nodes[i]);
				continue;
			}
			//cout << k << " - " << territory[k] << '\n';
			auto &vc = node[territory[split_nodes[i]]];
			vc.erase(remove(vc.begin(),vc.end(),split_nodes[i]),vc.end());
			node[territory[k]].push_back(split_nodes[i]);
			territory[split_nodes[i]] = territory[k];
		}	
		rebuild_node_lists();
		split_nodes = choosen;
		choosen.clear();
	
		for(int jj : split_nodes){
			int cur = territory[jj];
			if(cand[jj].empty()) continue;
			int bestk = -1;double bestCost = 1e18;bool picked = 0;
			// uu tiên t?ng kho?ng cách tru?c 
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
				// uu tiên m?c d? chênh l?ch don hàng, khách hàng 
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
		bool ok = 1;
		for(int j = 1;j <= p;j++){
			if(!connected(j)){
				ok = 0;
				//cout << "faill\n";
				break;
			}
		}
		if(!ok)break;
	}
}

void local_search(double gamma = gamma_,int limit_moves = 10000){
	rebuild_node_lists();
	double cur = meritfunction();
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
				double delta = gamma * Fdelta + (1 - gamma) * (Gafter - Gbefore);
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

bool print_output(pair<double,vector<int>> &x){
	map<int,int> mp;
	int cnt = 0;
	for(int i = 1;i <= p;i++)node[i].clear();
	for(int i = 0;i < x.Y.size();i++){
		if(!mp.count(x.Y[i])){
			cnt++;
			mp[x.Y[i]] = cnt;
			p_centroid[cnt] = x.Y[i];
		}
		node[mp[x.Y[i]]].push_back(i + 1);
		territory[i + 1] = mp[x.Y[i]];
		//cout << mp[x.Y[i]] << " " << i + 1<< '\n';
	}
	for(int k = 1;k <= p;k++){
		if(!connected(k)){
			(cout << "No solution found for the connected graph!\nHowever, this is the optimal solution for the above graph : \n");
		}
	}
	cout << "Accepted Range of orders for each territory : ";
	cout << (1 - t[0]) * u[0] << " " << (1 + t[0]) * u[0] << '\n';
	cout << "Accepted Range of customers for each territory : ";
	cout << (1 - t[1]) * u[1] << " " << (1 + t[1]) * u[1] << '\n';
	double d__ = 0.0;
	cnt = 0;
	for(int k = 1;k <= p;k++){
		cout << "Centroid " << k << " at node " << p_centroid[k] << ": ";
		int w_0 = 0;
		int w_1 = 0;
		double d_ = 0;
		double radius = 0;
		for(int v : node[k]){
			cout << v << " ";
			w_0 += w[v][0];
			w_1 += w[v][1];
			radius = max(radius,d[p_centroid[k]][v]);
			d_ += d[p_centroid[k]][v];
 		}
 		double x = max({w_0 - (1 + t[0]) * u[0],(1 - t[0]) * u[0] - w_0,0.0}) + max({w_1 - (1 + t[1]) * u[1],(1 - t[1]) * u[1] - w_1,0.0});
 		if(w_0 < (1 - t[0]) * u[0] || w_0 > (1 + t[0]) * u[0] || w_1 < (1 - t[1]) * u[1] || w_1 > (1 + t[1]) * u[1])cnt++;
 	
 		d__ += d_;
 		cout << "\n1 - Orders \t= " << w_0 << "\n2 - Customers \t= " << w_1 << "\n3 - Radius \t= " << radius << "\n4 - Distant \t= " << d_ << '\n'; 
	}
	cout << d__ << " - " << cnt << " - ";
	cout << meritfunction() << '\n';
	return 1;
}

unordered_map<int,int> Q;
pair<double,vector<int>> x,x_best;
int inter = 1,lim = 200;

signed main(){
	
	read();
	freopen("input//2DU60-05-1.dat","r",stdin);
	input();
//	cout << setprecision(10) << fixed << gamma_ << "\n";
	x_best.X = (double)1e18;
	bool f = 0;
	while(inter <= lim){
		if(f && inter > 100)break;
		x.X = 0;
		x.Y.clear();
		location();
		long long hash = 0;
		allocation();
		bool ok = 1;
		for(int i = 1;i <= p;i++)if(!connected(i)){ok = 0;break;}
		if(ok){
			f = 1;
			local_search();
		}
		for(int i = 1;i <= n;i++){
			x.Y.push_back(p_centroid[territory[i]]);
			hash = (hash * 23 +  x.Y.back()) % MOD;
		}
		if(Q.count(hash) && f)break;
		Q[hash] = 1;
		x.X = meritfunction() - ok * 2319;
		//cout << x.X << "\n";
		if(x_best.X > x.X){
			x_best.X = x.X;
			x_best.Y.clear();
			x_best.Y = x.Y;
			inter = 1;
		}else inter += 1;
	}
	//cout << f << " " << x_best.X << "\n";
	// if(f)cout << "hi\n";
	// cout << x_best.X << "\n";
	// for(auto v : x_best.Y)cout << v << " ";
	// cout << '\n';
	if(!f){
		cout << "No solution!";
		return 0;
	}
	print_output(x_best);
	
}