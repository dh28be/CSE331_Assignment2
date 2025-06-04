#include <bits/stdc++.h>
#define x first
#define y second

using namespace std;
using namespace chrono;

typedef long long ll;

const ll INF = LLONG_MAX;
const int SAMPLE_SIZE_LARGE = 10000;
const int SAMPLE_SIZE_SMALL = 20;
const int NUM_REPS = 10;

struct Node {
    int num;
    ll x, y;
};

struct Edgel {
    ll u, v, w;
};

struct Edged {
    ll u, v;
    double w;
};

struct Result {
    ll mst_based_l1;
    double mst_based_l2;
    ll held_karp_l1;
    double held_karp_l2;
    ll mo;
    ll hilbert_mo;
    ll time[6];
};

struct Results {
    Result org;
    Result sample_large;
    Result sample_small;
};

struct Dataset {
    string input_path;
    string name;
    int dimension;
    vector<Node> nodes;
    vector<Node> sample_large;
    vector<Node> sample_small;

    Dataset(string _input_path, string _name, int _dimension)
    : input_path(_input_path), name(_name), dimension(_dimension) {
        nodes.resize(dimension);
        sample_large.resize(min(dimension, SAMPLE_SIZE_LARGE));
        sample_small.resize(SAMPLE_SIZE_SMALL);
        read();
    }

    void read() {
        ifstream input(input_path);
        int metadata_lines;
        if(name == "a280") metadata_lines = 6;
        else if(name == "xql662") metadata_lines = 8;
        else if(name == "kz9976") metadata_lines = 7;
        else if(name == "mona_lisa") metadata_lines = 6;

        string buf;
        while(metadata_lines--) getline(input, buf);

        if(name != "kz9976") {
            for(int i = 0; i < dimension; ++i)
                input >> nodes[i].num >> nodes[i].x >> nodes[i].y;
        }
        else {
            double x, y;
            for(int i = 0; i < dimension; ++i) {
                input >> nodes[i].num >> x >> y;
                nodes[i].x = x*1000;
                nodes[i].y = y*1000;
            }
        }
        input.close();

        int sample_large_sz = sample_large.size();
        for(int i = 0; i < sample_large_sz; ++i)
            sample_large[i] = nodes[i];

        int sample_small_sz = sample_small.size();
        for(int i = 0; i < sample_small_sz; ++i)
            sample_small[i] = nodes[i];
    }
};

struct Experiment {
    static ll abs(ll x) {
        return x > 0 ? x : -x;
    }

    static ll sqr(ll x) {
        return x*x;
    }

    static ll l1_norm(Node &i, Node &j) {
        return abs(i.x-j.x) + abs(i.y-j.y);
    }

    static double l2_norm(Node &i, Node &j) {
        return sqrt(sqr(i.x-j.x) + sqr(i.y-j.y));
    }

    static ll held_karp_rec_l1(vector<vector<ll>> &dp, int cur, int visited, vector<Node> &nodes) {
        int sz = nodes.size();
        if(visited == (1 << sz) - 1) return l1_norm(nodes[cur], nodes[0]);
        if(dp[cur][visited]) return dp[cur][visited];
        dp[cur][visited] = INF;
        for(int i = 0; i < sz; ++i) {
            if(visited & (1 << i)) continue;
            ll cost = held_karp_rec_l1(dp, i, visited | (1 << i), nodes) + l1_norm(nodes[cur], nodes[i]);
            dp[cur][visited] = min(dp[cur][visited], cost);
        }
        return dp[cur][visited];
    }

    static ll held_karp_l1(vector<Node> &nodes) {
        int sz = nodes.size();
        vector<vector<ll>> dp(sz, vector<ll>(1 << sz, 0));
        return held_karp_rec_l1(dp, 0, 1, nodes);
    }

    static double held_karp_rec_l2(vector<vector<double>> &dp, int cur, int visited, vector<Node> &nodes) {
        int sz = nodes.size();
        if(visited == (1 << sz) - 1) return l2_norm(nodes[cur], nodes[0]);
        if(dp[cur][visited]) return dp[cur][visited];
        dp[cur][visited] = INF;
        for(int i = 0; i < sz; ++i) {
            if(visited & (1 << i)) continue;
            double cost = held_karp_rec_l2(dp, i, visited | (1 << i), nodes) + l2_norm(nodes[cur], nodes[i]);
            dp[cur][visited] = min(dp[cur][visited], cost);
        }
        return dp[cur][visited];
    }

    static double held_karp_l2(vector<Node> &nodes) {
        int sz = nodes.size();
        vector<vector<double>> dp(sz, vector<double>(1 << sz, 0));
        return held_karp_rec_l2(dp, 0, 1, nodes);
    }

    struct UnionFind {
        vector<int> par;

        UnionFind(int sz) {
            par.resize(sz);
            for(int i = 0; i < sz; ++i)
                par[i] = i;
        }

        int root(int x) {
            if(x == par[x]) return x;
            return par[x] = root(par[x]);
        }

        bool merge(int x, int y) {
            x = root(x), y = root(y);
            if(x == y) return 0;
            par[x] = y;
            return 1;
        }
    };

    static void find_euler_tour(vector<vector<int>> &adj, vector<bool> &visited, vector<int> &euler_tour, int cur) {
        visited[cur] = 1;
        euler_tour.push_back(cur);

        bool flag = 0;
        for(auto &it : adj[cur]) {
            if(visited[it]) continue;
            flag = 1;
            find_euler_tour(adj, visited, euler_tour, it);
        }

        if(flag) euler_tour.push_back(cur);
    }

    static vector<Edgel> construct_mst_l1(vector<Node> &nodes) {
        int sz = nodes.size();
        UnionFind uf(sz);
        vector<Edgel> edges, ret;

        for(int i = 0; i < sz; ++i)
        for(int j = i+1; j < sz; ++j)
            edges.push_back({i, j, l1_norm(nodes[i], nodes[j])});
        auto cmp = [&](Edgel &i, Edgel &j) {
            return i.w < j.w;
        };
        sort(edges.begin(), edges.end(), cmp);

        int idx = 0;
        while(int(ret.size()) < sz-1) {
            if(uf.merge(edges[idx].u, edges[idx].v)) ret.push_back(edges[idx]);
            idx++;
        }

        return ret;
    }

    static ll mst_based_l1(vector<Node> &nodes) {
        int sz = nodes.size();
        vector<Edgel> mst = construct_mst_l1(nodes);

        vector<vector<int>> adj(sz);
        for(auto &it : mst) {
            adj[it.u].push_back(it.v);
            adj[it.v].push_back(it.u);
        }

        vector<bool> visited(sz, 0);
        vector<int> euler_tour;
        find_euler_tour(adj, visited, euler_tour, 0);

        int s = euler_tour[0];
        int len = euler_tour.size();
        int ret = 0;
        fill(visited.begin(), visited.end(), 0);
        for(int i = 1; i < len; ++i) {
            int cur = euler_tour[i];
            if(visited[cur]) continue;
            visited[cur] = 1;
            ret += l1_norm(nodes[s], nodes[cur]);
            s = cur;
        }

        return ret;
    }

    static vector<Edged> construct_mst_l2(vector<Node> &nodes) {
        int sz = nodes.size();
        UnionFind uf(sz);
        vector<Edged> edges, ret;

        for(int i = 0; i < sz; ++i)
        for(int j = i+1; j < sz; ++j)
            edges.push_back({i, j, l2_norm(nodes[i], nodes[j])});
        auto cmp = [&](Edged &i, Edged &j) {
            return i.w < j.w;
        };
        sort(edges.begin(), edges.end(), cmp);

        int idx = 0;
        while(int(ret.size()) < sz-1) {
            if(uf.merge(edges[idx].u, edges[idx].v)) ret.push_back(edges[idx]);
            idx++;
        }

        return ret;
    }

    static double mst_based_l2(vector<Node> &nodes) {
        int sz = nodes.size();
        vector<Edged> mst = construct_mst_l2(nodes);

        vector<vector<int>> adj(sz);
        for(auto &it : mst) {
            adj[it.u].push_back(it.v);
            adj[it.v].push_back(it.u);
        }

        vector<bool> visited(sz, 0);
        vector<int> euler_tour;
        find_euler_tour(adj, visited, euler_tour, 0);

        int s = euler_tour[0];
        int len = euler_tour.size();
        double ret = 0;
        fill(visited.begin(), visited.end(), 0);
        for(int i = 1; i < len; ++i) {
            int cur = euler_tour[i];
            if(visited[cur]) continue;
            visited[cur] = 1;
            ret += l2_norm(nodes[s], nodes[cur]);
            s = cur;
        }

        return ret;
    }

    static ll mo(vector<Node> &nodes) {
        int sz = nodes.size();
        ll mxx = 0, mnx = INF;
        ll mxy = 0, mny = INF;
        for(auto &it : nodes) {
            mxx = max(mxx, it.x);
            mnx = min(mnx, it.x);
            mxy = max(mxy, it.y);
            mny = min(mny, it.y);
        }

        ll nx = mxx-mnx;
        ll ny = mxy-mny;
        ll bucketx = sqrt(nx);
        if(!bucketx) bucketx++;
        ll buckety = sqrt(ny);
        if(!buckety) buckety++;

        vector<Node> nodesx = nodes;
        vector<Node> nodesy = nodes;
        auto cmpx = [&](Node &i, Node &j) {
            if(i.x/bucketx == j.x/bucketx) return i.y < j.y;
            return i.x < j.x;
        };
        auto cmpy = [&](Node &i, Node &j) {
            if(i.x/buckety == j.x/buckety) return i.y < j.y;
            return i.x < j.x;
        };
        sort(nodesx.begin(), nodesx.end(), cmpx);
        sort(nodesy.begin(), nodesy.end(), cmpy);

        ll ansx = 0, ansy = 0;
        for(int i = 0; i < sz; ++i) {
            int ni = (i+1)%sz;
            ansx += l1_norm(nodesx[i], nodesx[ni]);
            ansy += l1_norm(nodesy[i], nodesy[ni]);
        }

        return min(ansx, ansy);
    }

    static ll hilbert_mo(vector<Node> &nodes) {
        int sz = nodes.size();
        ll mxx = 0, mnx = INF;
        ll mxy = 0, mny = INF;
        for(auto &it : nodes) {
            mxx = max(mxx, it.x);
            mnx = min(mnx, it.x);
            mxy = max(mxy, it.y);
            mny = min(mny, it.y);
        }

        ll nx = mxx-mnx;
        ll ny = mxy-mny;
        ll bucketx = nx / sqrt(sz);
        if(!bucketx) bucketx++;
        ll buckety = ny / sqrt(sz);
        if(!buckety) buckety++;

        vector<Node> nodesx = nodes;
        vector<Node> nodesy = nodes;
        auto cmpx = [&](Node &i, Node &j) {
            if(i.x/bucketx == j.x/bucketx) return i.y < j.y;
            return i.x < j.x;
        };
        auto cmpy = [&](Node &i, Node &j) {
            if(i.x/buckety == j.x/buckety) return i.y < j.y;
            return i.x < j.x;
        };
        sort(nodesx.begin(), nodesx.end(), cmpx);
        sort(nodesy.begin(), nodesy.end(), cmpy);

        ll ansx = 0, ansy = 0;
        for(int i = 0; i < sz; ++i) {
            int ni = (i+1)%sz;
            ansx += l1_norm(nodesx[i], nodesx[ni]);
            ansy += l1_norm(nodesy[i], nodesy[ni]);
        }

        return min(ansx, ansy);
    }

    static Results conduct_exp(vector<Node> &org, vector<Node> &sample_large, vector<Node> &sample_small) {
        Result org_res;
        Result sample_large_res;
        Result sample_small_res;
        system_clock::time_point start_time;
        system_clock::time_point end_time;
        int sum_time;
        bool is_kz9976 = org.size() == 9976;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            org_res.mo = mo(org);
            end_time = system_clock::now();
            if(is_kz9976) org_res.mo /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        org_res.time[4] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            org_res.hilbert_mo = hilbert_mo(org);
            end_time = system_clock::now();
            if(is_kz9976) org_res.hilbert_mo /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        org_res.time[5] = sum_time/NUM_REPS;


        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_large_res.mst_based_l1 = mst_based_l1(sample_large);
            end_time = system_clock::now();
            if(is_kz9976) sample_large_res.mst_based_l1 /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_large_res.time[2] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_large_res.mst_based_l2 = mst_based_l2(sample_large);
            end_time = system_clock::now();
            if(is_kz9976) sample_large_res.mst_based_l2 /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_large_res.time[3] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_large_res.mo = mo(sample_large);
            end_time = system_clock::now();
            if(is_kz9976) sample_large_res.mo /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_large_res.time[4] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_large_res.hilbert_mo = hilbert_mo(sample_large);
            end_time = system_clock::now();
            if(is_kz9976) sample_large_res.hilbert_mo /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_large_res.time[5] = sum_time/NUM_REPS;


        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_small_res.held_karp_l1 = held_karp_l1(sample_small);
            end_time = system_clock::now();
            if(is_kz9976) sample_small_res.held_karp_l1 /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_small_res.time[0] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_small_res.held_karp_l2 = held_karp_l2(sample_small);
            end_time = system_clock::now();
            if(is_kz9976) sample_small_res.held_karp_l2 /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_small_res.time[1] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_small_res.mst_based_l1 = mst_based_l1(sample_small);
            end_time = system_clock::now();
            if(is_kz9976) sample_small_res.mst_based_l1 /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_small_res.time[2] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_small_res.mst_based_l2 = mst_based_l2(sample_small);
            end_time = system_clock::now();
            if(is_kz9976) sample_small_res.mst_based_l2 /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_small_res.time[3] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_small_res.mo = mo(sample_small);
            end_time = system_clock::now();
            if(is_kz9976) sample_small_res.mo /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_small_res.time[4] = sum_time/NUM_REPS;

        sum_time = 0;
        for(int i = 0; i < NUM_REPS; ++i) {
            start_time = system_clock::now();
            sample_small_res.hilbert_mo = hilbert_mo(sample_small);
            end_time = system_clock::now();
            if(is_kz9976) sample_small_res.hilbert_mo /= 1000.0;
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_small_res.time[5] = sum_time/NUM_REPS;

        return {org_res, sample_large_res, sample_small_res};
    }

    static void write(ofstream &output, Results res, string name) {
        const int WIDTH = 50;
        string line1, line2;
        string algorithms[6] = {
            "Held-Karp L1",
            "Held-Karp L2",
            "MST_based L1",
            "MST_based L2",
            "mo's",
            "Hilbert MO"
        };

        for(int i = 0; i < WIDTH; ++i) {
            line1.push_back('-');
            line2.push_back('=');
        }

        int left = (WIDTH-name.length()-2)/2;
        int right = (WIDTH-name.length()-2)/2 + (WIDTH-name.length()-2)%2;
        output << line2.substr(0, left)+' ' << name << ' '+line2.substr(0, right) << endl;

        output << endl;
        output << "Original" << endl;
        output << endl;
        output << algorithms[4] << ": " << res.org.mo << ", " << res.org.time[4] << "ms" << endl;
        output << algorithms[5] << ": " << res.org.hilbert_mo << ", " << res.org.time[5] << "ms" << endl;
        output << endl;
        output << line1 << endl;

        output << endl;
        output << to_string(SAMPLE_SIZE_LARGE)+" Samples" << endl;
        output << endl;
        output << algorithms[2] << ": " << res.sample_large.mst_based_l1 << ", " << res.sample_large.time[2] << "ms" << endl;
        output << algorithms[3] << ": " << res.sample_large.mst_based_l2 << ", " << res.sample_large.time[3] << "ms" << endl;
        output << algorithms[4] << ": " << res.sample_large.mo << ", " << res.sample_large.time[4] << "ms" << endl;
        output << algorithms[5] << ": " << res.sample_large.hilbert_mo << ", " << res.sample_large.time[5] << "ms" << endl;
        output << endl;
        output << line1 << endl;

        output << endl;
        output << to_string(SAMPLE_SIZE_SMALL)+" Samples" << endl;
        output << endl;
        output << algorithms[0] << ": " << res.sample_small.held_karp_l1 << ", " << res.sample_small.time[0] << "ms" << endl;
        output << algorithms[1] << ": " << res.sample_small.held_karp_l2 << ", " << res.sample_small.time[1] << "ms" << endl;
        output << algorithms[2] << ": " << res.sample_small.mst_based_l1 << ", " << res.sample_small.time[2] << "ms" << endl;
        output << algorithms[3] << ": " << res.sample_small.mst_based_l2 << ", " << res.sample_small.time[3] << "ms" << endl;
        output << algorithms[4] << ": " << res.sample_small.mo << ", " << res.sample_small.time[4] << "ms" << endl;
        output << algorithms[5] << ": " << res.sample_small.hilbert_mo << ", " << res.sample_small.time[5] << "ms" << endl;
        output << endl;

        output << line2 << endl;
        output << endl;
        output << endl;
    }
};

struct GT {
    string input_path;
    string name;
    int dimension;
    vector<int> tour;

    GT(string _input_path, string _name, int _dimension)
    : input_path(_input_path), name(_name), dimension(_dimension) {
        tour.resize(dimension);
        read();
    }

    void read() {
        ifstream input(input_path);
        int metadata_lines;
        if(name == "a280") metadata_lines = 4;
        else metadata_lines = 5;

        string buf;
        while(metadata_lines--) getline(input, buf);

        for(int i = 0; i < dimension; ++i)
            input >> tour[i];
        input.close();
    }

    void eval(vector<Node> &nodes, ofstream &output, string name) {
        int sz = tour.size();
        pair<ll, double> res = {0, 0};
        for(int i = 0; i < sz; ++i) {
            int ni = (i+1)%sz;
            res.x += Experiment::l1_norm(nodes[tour[i]-1], nodes[tour[ni]-1]);
            res.y += Experiment::l2_norm(nodes[tour[i]-1], nodes[tour[ni]-1]);
        }

        if(name != "kz9976") {
            output << name+"_L1: " << res.x << endl;
            output << name+"_L2: " << res.y << endl;
        }
        else {
            output << name+"_L1: " << res.x/1000.0 << endl;
            output << name+"_L2: " << res.y/1000.0 << endl;
        }
    }
};

int main() {
    Dataset a280("datasets/a280.tsp", "a280", 280);
    Dataset xql662("datasets/xql662.tsp", "xql662", 662);
    Dataset kz9976("datasets/kz9976.tsp", "kz9976", 9976);
    Dataset mona_lisa("datasets/mona-lisa100K.tsp", "mona_lisa", 100000);

    GT a280_gt("datasets/a280.opt.tour", "a280", 280);
    GT xql662_gt("datasets/xql662.opt.tour", "xql662", 662);
    GT kz9976_gt("datasets/kz9976.opt.tour", "kz9976", 9976);
    GT mona_lisa_gt("datasets/mona-lisa100K.opt.tour", "mona_lisa", 100000);

    ofstream output("results.txt");
    output << fixed;
    const int WIDTH = 50;
    string line1, line2;
    string gt_str = "Ground Truth";

    for(int i = 0; i < WIDTH; ++i) {
        line1.push_back('-');
        line2.push_back('=');
    }

    int left = (WIDTH-gt_str.length()-2)/2;
    int right = (WIDTH-gt_str.length()-2)/2 + (WIDTH-gt_str.length()-2)%2;
    output << line2.substr(0, left)+' ' << gt_str << ' '+line2.substr(0, right) << endl;

    output << endl;
    a280_gt.eval(a280.nodes, output, "a280");
    xql662_gt.eval(xql662.nodes, output, "xql662");
    kz9976_gt.eval(kz9976.nodes, output, "kz9976");
    mona_lisa_gt.eval(mona_lisa.nodes, output, "mona_lisa");
    output << endl;

    output << line2 << endl;
    output << endl;
    output << endl;

    Experiment::write(output, Experiment::conduct_exp(a280.nodes, a280.sample_large, a280.sample_small), "a280");
    Experiment::write(output, Experiment::conduct_exp(xql662.nodes, xql662.sample_large, xql662.sample_small), "xql662");
    Experiment::write(output, Experiment::conduct_exp(kz9976.nodes, kz9976.sample_large, kz9976.sample_small), "kz9976");
    Experiment::write(output, Experiment::conduct_exp(mona_lisa.nodes, mona_lisa.sample_large, mona_lisa.sample_small), "mona_lisa");
    output.close();

    return 0;
}