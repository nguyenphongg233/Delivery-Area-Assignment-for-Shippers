#include<bits/stdc++.h>

using namespace std;

#define read() ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define day() time_t now = time(0);char* x = ctime(&now);cerr<<"Right Now Is : "<<x<<"\n"

#define ii pair<int,int>
#define X first1
#define Y second 

const long long MAX = (int)1e6 + 5;
const long long INF = (int)1e9;
const long long MOD = (int)1e9 + 7;

int l,r;
string s;
struct node{
	int l,r;
	string s;
};
signed main(){
	
	read();

	string c = "local_search.exe";
	vector<node> f;	
//	f.push_back({1,20,"2DU60-05-"});
//	f.push_back({1,20,"2DU80-05-"});
//	f.push_back({1,20,"2DU100-05-"});
//	f.push_back({10,20,"2DU120-05-"});
//	f.push_back({8,10,"DU150-05-"});	
	f.push_back({1,10,"DU200-05-"});
//	f.push_back({11,11,"2DU80-05-"});
//	f.push_back({19,19,"2DU80-05-"});
//	f.push_back({4,10,"2DU100-05-"});
//	f.push_back({13,14,"2DU100-05-"});
//	f.push_back({16,16,"2DU100-05-"});
//	f.push_back({18,20,"2DU100-05-"});	
//	f.push_back({1,20,"2DU120-05-"});
//	f.push_back({3,10,"DU150-05-"});
//	f.push_back({5,10,"DU200-05-"});
//	f.push_back({1,1,"DU280-05-"});
	
	//s = to_string(t + 1) + "." + s;
	
	string dictionary = "Local_Search";
	system(("md " + dictionary).c_str());
	
	for(int j = 0;j < (int)f.size();j++){
		int l = f[j].l;
		int r = f[j].r;
		string s = f[j].s;
		for(int i = l;i <= r;i++){
			
			string t = "";
			if(i < 10)t = "0";
			cerr << (c + " < input\\" + s + to_string(i) + ".dat > " + dictionary + "\\" + s + t + to_string(i) + ".out").c_str() << "\n";
			system((c + " < input\\" + s + to_string(i) + ".dat > " + dictionary + "\\" + s + t + to_string(i) + ".out").c_str());
			//system(("type " + s + to_string(i) + ".out >>" + "result.txt").c_str());
		}
	}
	
	system("run.bat");
//	freopen("data.txt","w",stdout);
//	cout << t + 1;
}

