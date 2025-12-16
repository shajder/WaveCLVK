/*
MIT License

Copyright (c) 2025 Marcin Hajder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "wave_app.hpp"
#include <iostream>
#include <boost/program_options.hpp>

int main(int argc, char** argv)
{
    WaveApp app;

    boost::program_options::options_description desc("Program options");
    desc.add_options()("help,h", "show help")(
        "width,w",
        boost::program_options::value<size_t>(&app.opts.window_width)
            ->default_value(1024),
        "window width")(
        "height,h",
        boost::program_options::value<size_t>(&app.opts.window_height)
            ->default_value(1024),
        "window height")(
        "technique,t",
        boost::program_options::value<unsigned short>(&app.opts.technique)
            ->default_value(0),
        "spectrum technique (0 - Phillips, 1 - Jonswap)")(
        "foam,f",
        boost::program_options::value<unsigned short>(&app.opts.foam_technique)
            ->default_value(0),
        "foam technique (0 - default, 1 - Experimental, CFD based)")(
        "platform,p",
        boost::program_options::value<unsigned short>(&app.opts.plat_index)
            ->default_value(0),
        "platform index")(
        "device,d",
        boost::program_options::value<unsigned short>(&app.opts.dev_index)
            ->default_value(0),
        "device index");

    try {

        boost::program_options::variables_map vm;
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        // Obsługa opcji pomocy
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

    } catch (const boost::program_options::unknown_option& e) {
        std::cerr << "Unknown option: " << e.what() << std::endl;
        return 1;
    } catch (const boost::program_options::invalid_option_value& e) {
        std::cerr << "invalid value for option: " << e.what() << std::endl;
        return 1;
    } catch (const boost::program_options::required_option& e) {
        std::cerr << "required option is missing: " << e.what() << std::endl;
        return 1;
    } catch (const boost::program_options::error& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    try
    {
        app.run();
    } catch (const std::exception& e)
    {
        fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
