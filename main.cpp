#include <bits/stdc++.h>
#define x first
#define y second

using namespace std;
using namespace chrono;

typedef long long ll;

const ll INF = LLONG_MAX;
const int SAMPLE_SIZE = 20;

struct Node {
    int num;
    ll x, y;
};

struct Result {
    ll christodifes_heuristic_l1;
    double christodifes_heuristic_l2;
    ll held_karp_l1;
    double held_karp_l2;
    ll mo;
    ll hilbert_mo;
    ll time[6];
};
typedef pair<Result, Result> prr;

struct Dataset {
    string input_path;
    string name;
    int dimension;
    vector<Node> nodes;
    vector<Node> sample;

    Dataset(string _input_path, string _name, int _dimension)
    : input_path(_input_path), name(_name), dimension(_dimension) {
        nodes.resize(dimension);
        sample.resize(SAMPLE_SIZE);
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
                nodes[i].x = x*10000;
                nodes[i].y = y*10000;
                nodes[i].x /= 10;
                nodes[i].y /= 10;
            }
        }
        input.close();

        for(int i = 0; i < SAMPLE_SIZE; ++i)
            sample[i] = nodes[i];
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

    static prr conduct_exp(vector<Node> &org, vector<Node> &sample) {
        Result org_res;
        Result sample_res;
        system_clock::time_point start_time;
        system_clock::time_point end_time;
        int sum_time;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            // org_res.christodifes_heuristic_l1 = christodifes_heuristic_l1(org);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        org_res.time[2] = sum_time/10;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            // org_res.christodifes_heuristic_l2 = christodifes_heuristic_l2(org);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        org_res.time[3] = sum_time/10;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            org_res.mo = mo(org);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        org_res.time[4] = sum_time/10;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            org_res.hilbert_mo = hilbert_mo(org);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        org_res.time[5] = sum_time/10;


        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            sample_res.held_karp_l1 = held_karp_l1(sample);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_res.time[0] = sum_time/10;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            sample_res.held_karp_l2 = held_karp_l2(sample);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_res.time[1] = sum_time/10;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            // sample_res.christodifes_heuristic_l1 = christodifes_heuristic_l1(sample);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_res.time[2] = sum_time/10;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            // sample_res.christodifes_heuristic_l2 = christodifes_heuristic_l2(sample);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_res.time[3] = sum_time/10;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            sample_res.mo = mo(sample);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_res.time[4] = sum_time/10;

        sum_time = 0;
        for(int i = 0; i < 10; ++i) {
            start_time = system_clock::now();
            sample_res.hilbert_mo = hilbert_mo(sample);
            end_time = system_clock::now();
            sum_time += duration_cast<milliseconds>(end_time-start_time).count();
        }
        sample_res.time[5] = sum_time/10;

        return {org_res, sample_res};
    }

    static void write(ofstream &output, prr res, string name) {
        const int WIDTH = 50;
        string line1, line2;
        string algorithms[6] = {"Held-Karp L1",
            "Held-Karp L2",
            "Christofides Heuristic L1",
            "Christofides Heuristic L2",
            "mo's",
            "Hilbert MO"};

        for(int i = 0; i < WIDTH; ++i) {
            line1.push_back('-');
            line2.push_back('=');
        }

        int left = (WIDTH-name.length()-2)/2;
        int right = (WIDTH-name.length()-2)/2 + (WIDTH-name.length()-2)%2;
        output << line2.substr(0, left)+' ' << name << ' '+line2.substr(0, right) << endl;

        if(name != "kz9976") {
            output << endl;
            output << "Original" << endl;
            output << endl;
            output << algorithms[2] << ": " << res.x.christodifes_heuristic_l1 << ", " << res.x.time[2] << "ms" << endl;
            output << algorithms[3] << ": " << res.x.christodifes_heuristic_l2 << ", " << res.x.time[3] << "ms" << endl;
            output << algorithms[4] << ": " << res.x.mo << ", " << res.x.time[4] << "ms" << endl;
            output << algorithms[5] << ": " << res.x.hilbert_mo << ", " << res.x.time[5] << "ms" << endl;
            output << endl;
            output << line1 << endl;

            output << endl;
            output << to_string(SAMPLE_SIZE)+" Samples" << endl;
            output << endl;
            output << algorithms[0] << ": " << res.y.held_karp_l1 << ", " << res.y.time[0] << "ms" << endl;
            output << algorithms[1] << ": " << res.y.held_karp_l2 << ", " << res.y.time[1] << "ms" << endl;
            output << algorithms[2] << ": " << res.y.christodifes_heuristic_l1 << ", " << res.y.time[2] << "ms" << endl;
            output << algorithms[3] << ": " << res.y.christodifes_heuristic_l2 << ", " << res.y.time[3] << "ms" << endl;
            output << algorithms[4] << ": " << res.y.mo << ", " << res.y.time[4] << "ms" << endl;
            output << algorithms[5] << ": " << res.y.hilbert_mo << ", " << res.y.time[5] << "ms" << endl;
            output << endl;
        }
        else {
            output << endl;
            output << "Original" << endl;
            output << endl;
            output << algorithms[2] << ": " << res.x.christodifes_heuristic_l1/1000.0 << ", " << res.x.time[2] << "ms" << endl;
            output << algorithms[3] << ": " << res.x.christodifes_heuristic_l2/1000.0 << ", " << res.x.time[3] << "ms" << endl;
            output << algorithms[4] << ": " << res.x.mo/1000.0 << ", " << res.x.time[4] << "ms" << endl;
            output << algorithms[5] << ": " << res.x.hilbert_mo/1000.0 << ", " << res.x.time[5] << "ms" << endl;
            output << endl;
            output << line1 << endl;

            output << endl;
            output << "20 Samples" << endl;
            output << endl;
            output << algorithms[0] << ": " << res.y.held_karp_l1/1000.0 << ", " << res.y.time[0] << "ms" << endl;
            output << algorithms[1] << ": " << res.y.held_karp_l2/1000.0 << ", " << res.y.time[1] << "ms" << endl;
            output << algorithms[2] << ": " << res.y.christodifes_heuristic_l1/1000.0 << ", " << res.y.time[2] << "ms" << endl;
            output << algorithms[3] << ": " << res.y.christodifes_heuristic_l2/1000.0 << ", " << res.y.time[3] << "ms" << endl;
            output << algorithms[4] << ": " << res.y.mo/1000.0 << ", " << res.y.time[4] << "ms" << endl;
            output << algorithms[5] << ": " << res.y.hilbert_mo/1000.0 << ", " << res.y.time[5] << "ms" << endl;
            output << endl;
        }

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
        if(name != "kz9976") {
            pair<ll, double> res = {0, 0};
            for(int i = 0; i < sz; ++i) {
                int ni = (i+1)%sz;
                res.x += Experiment::l1_norm(nodes[tour[i]-1], nodes[tour[ni]-1]);
                res.y += Experiment::l2_norm(nodes[tour[i]-1], nodes[tour[ni]-1]);
            }

            output << name+"_L1: " << res.x << endl;
            output << name+"_L2: " << res.y << endl;
        }
        else {
            pair<double, double> res = {0, 0};
            for(int i = 0; i < sz; ++i) {
                int ni = (i+1)%sz;
                res.x += Experiment::l1_norm(nodes[tour[i]-1], nodes[tour[ni]-1])/1000.0;
                res.y += Experiment::l2_norm(nodes[tour[i]-1], nodes[tour[ni]-1])/1000.0;
            }

            output << name+"_L1: " << res.x << endl;
            output << name+"_L2: " << res.y << endl;
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

    Experiment::write(output, Experiment::conduct_exp(a280.nodes, a280.sample), "a280");
    Experiment::write(output, Experiment::conduct_exp(xql662.nodes, xql662.sample), "xql662");
    Experiment::write(output, Experiment::conduct_exp(kz9976.nodes, kz9976.sample), "kz9976");
    Experiment::write(output, Experiment::conduct_exp(mona_lisa.nodes, mona_lisa.sample), "mona_lisa");
    output.close();

    return 0;
}